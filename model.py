import pandas as pd
import numpy as np
import torch
import faiss
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import matplotlib.pyplot as plt  # Added import
import networkx as nx  # Added import

# ----------------------------------------------------------------------------------------------------
# 1. Memory-Optimized Data Loading (Unchanged)
def load_data(sample_frac=0.3):
    """Stratified sampling with safe data types"""
    dtypes = {
        'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'C1': 'float32', 'C2': 'float32', 'C3': 'float32',
        'ProductCD': 'category',
        'card4': str,
        'P_emaildomain': str,
        'isFraud': 'int8'
    }

    try:
        transactions = pd.read_csv(
            'train_transaction.csv',
            dtype=dtypes,
            usecols=list(dtypes.keys())
        )

        fraud = transactions[transactions['isFraud'] == 1].sample(frac=sample_frac)
        non_fraud_count = int(len(fraud)/0.035*0.965)
        non_fraud = transactions[transactions['isFraud'] == 0].sample(n=non_fraud_count)

        return pd.concat([fraud, non_fraud]).sample(frac=1).reset_index(drop=True)

    except Exception as e:
        print(f"Data loading failed: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------------------------------------
# 2. Feature Engineering (Unchanged)
def preprocess_features(df, scaler=None, pca=None, fit_mode=False):
    """Process features with isolation between train/val/test data.

    Args:
        df: Input DataFrame (already split into train/val/test)
        scaler: Pre-fitted StandardScaler (required if fit_mode=False)
        pca: Pre-fitted PCA (required if fit_mode=False)
        fit_mode: True for training data (fit new scalers/PCA)

    Returns:
        Processed features tensor + scaler/pca (if fit_mode=True)
    """
    df = df.copy()
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    df['card4'] = df['card4'].fillna('unknown')

    # Convert TransactionDT to hourly bins
    df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24
    df = df.sort_values('TransactionDT')  # Time since last transaction (global)
    df['TimeSinceLast'] = df['TransactionDT'].diff().fillna(0)

    # 3. Numerical features (now includes temporal features)
    num_cols = ['TransactionAmt', 'C1', 'C2', 'C3',
                'TransactionHour', 'TimeSinceLast']
    num_features = df[num_cols].fillna(0)

    if fit_mode:
        scaler = StandardScaler().fit(num_features)
        num_scaled = scaler.transform(num_features)
    else:
        num_scaled = scaler.transform(num_features)

    email_hash = pd.get_dummies(pd.util.hash_pandas_object(df['P_emaildomain']) % 50, prefix='email')
    card_hash = pd.get_dummies(pd.util.hash_pandas_object(df['card4']) % 20, prefix='card')

    features = np.hstack([num_scaled, email_hash, card_hash])

    if fit_mode:
        pca = PCA(n_components=min(32, features.shape[1]))
        features = pca.fit_transform(features)
        return torch.tensor(features, dtype=torch.float32), scaler, pca
    else:
        features = pca.transform(features)
        return torch.tensor(features, dtype=torch.float32)

# ----------------------------------------------------------------------------------------------------
# 3. New: Semantic Similarity Edge Construction (Equation 1)
def build_semantic_similarity_edges(features, original_indices, threshold=0.8, batch_size=8192):
    """FAISS-accelerated edge construction"""
    edge_index = []
    num_nodes = features.size(0)
    features_np = features.cpu().numpy()

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(features_np)
    index = faiss.IndexFlatIP(features_np.shape[1])
    print("FAISS indexing started")
    index.add(features_np)

    for i in range(0, num_nodes, batch_size):
        batch = features_np[i:i+batch_size]
        similarities, neighbors = index.search(batch, 100)  # Top 100 neighbors

        for idx_in_batch, (sim_row, nbr_row) in enumerate(zip(similarities, neighbors)):
            # Map subset index to original dataset index
            subset_src = i + idx_in_batch
            original_src = original_indices[subset_src].item()

            # Filter valid neighbors within threshold
            valid = sim_row > threshold
            for subset_dst in nbr_row[valid]:
                if subset_src != subset_dst:
                    original_dst = original_indices[subset_dst].item()
                    edge_index.append([original_src, original_dst])
                    edge_index.append([original_dst, original_src])

    print(f"Generated {len(edge_index)//2} edges (FAISS)")
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long).to(features.device)

    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_tensor.unique(dim=1).to(features.device)

# ----------------------------------------------------------------------------------------------------
# 4. Imbalance Handling (Unchanged)
class MCD(nn.Module):
    def __init__(self, input_dim, gamma=0.01): # Adjust gamma
        super().__init__()
        self.gamma = gamma
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X_majority):
        scores = torch.sigmoid(self.selector(X_majority)).squeeze()
        drop_probs = self.gamma * (1 - scores)
        keep_mask = torch.bernoulli(1 - drop_probs)
        return torch.where(keep_mask)[0]

class AdaptiveMCD(MCD):
    def __init__(self, input_dim, fraud_ratio, alpha=0.5):
        gamma = (1 - fraud_ratio) * alpha
        super().__init__(input_dim, gamma=gamma)
        self.alpha = alpha

# ----------------------------------------------------------------------------------------------------
# 5. RL Agent and Policy
class SubgraphPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 actions: RW, K-hop, K-ego
        )

    def forward(self, x):
        logits = self.actor(x)
        return F.softmax(logits, dim=-1), logits

class RLAgent:
    def __init__(self, feat_dim, global_to_local=None):
        self.policy = SubgraphPolicy(feat_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.mces = LiteMCES()
        self.global_to_local = global_to_local

    def train_rl(self, nodes, features, edge_index, y_labels, n_epochs=50):
        """Batched RL training with average loss tracking"""
        device = features.device
        y_labels = y_labels.to(device)
        subset = nodes[torch.randperm(len(nodes))[:int(0.1*len(nodes))]].to(device)

        # Precompute adjacency for RLAgent's MCES instance
        self.mces._precompute_adjacency(edge_index.to(device))

        # Loss tracking
        epoch_losses = []

        for epoch in range(n_epochs):
            # Forward pass (batched)
            node_features = features[subset]
            probs, logits = self.policy(node_features)
            actions = torch.multinomial(probs, 1).squeeze()

            # Reward calculation
            rewards = []
            for node, action in zip(subset, actions):
                method_nodes = self.mces._execute_method(node.item(), action)
                method_nodes_tensor = torch.tensor(method_nodes, dtype=torch.long, device=device)
                method_nodes_local = self.global_to_local[method_nodes_tensor]
                valid = method_nodes_local != -1
                if valid.any():
                    reward = torch.log1p(y_labels[method_nodes_local[valid]].sum().float())
                else:
                    reward = torch.tensor(0.0, device=device)

                rewards.append(reward)

            rewards = torch.stack(rewards)

            # Policy gradient loss
            loss = -(torch.log(probs[range(len(subset)), actions]) * rewards).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track average loss
            epoch_losses.append(loss.item())

            if (epoch+1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if epoch >= 10 else loss.item()
                print(f"RL Epoch {epoch+1}: Loss={avg_loss:.4f}, Avg Reward={rewards.mean().item():.4f}")

# ----------------------------------------------------------------------------------------------------
# 6. Minor-node-centered explored Subgraph (MCES)
class LiteMCES:
    def __init__(self, k_rw=10, k_hop=3, k_ego=2):
        self.k_rw = k_rw  # Random walk length (paper: 10)
        self.k_hop = k_hop  # K-hop neighbors (paper: 3)
        self.k_ego = k_ego  # K-ego depth (paper: 2)
        self.max_neighbors = 1000  # Truncation limit
        self.adj_matrix = None

    def _precompute_adjacency(self, edge_index):
        """Convert edge_index to dense adjacency list on GPU"""
        num_nodes = edge_index.max().item() + 1

        # Initialize padded adjacency matrix
        self.adj_matrix = torch.full((num_nodes, self.max_neighbors), -1,
                                   dtype=torch.long, device=edge_index.device)

        # Get unique neighbors per node
        nodes, counts = torch.unique(edge_index[0], return_counts=True)
        for node, count in zip(nodes, counts):
            neighbors = edge_index[1][edge_index[0] == node][:self.max_neighbors]
            self.adj_matrix[node, :len(neighbors)] = neighbors

    def enhance_subgraph(self, edge_index, fraud_nodes, features, rl_agent=None, y_labels=None):
        if self.adj_matrix is None:
            self._precompute_adjacency(edge_index)
        sub_edges = []
        device = features.device
        fraud_nodes = fraud_nodes.to(device)

        if rl_agent:  # RL-enhanced path
            # Training handled internally by RLAgent
            rl_agent.train_rl(fraud_nodes, features, edge_index, y_labels)

            # Inference using trained policy
            for node in fraud_nodes:
                node = node.to(device)
                with torch.no_grad():
                    action_probs = rl_agent.policy(features[node])[0]
                    action = action_probs.argmax().item()

                # Single method execution based on RL choice
                method_nodes = self._execute_method(node, action)
                self._add_edges(sub_edges, node, method_nodes)

        else:  # Original three-method path
            for node in fraud_nodes:
                # Execute all three methods
                for method_id in [0, 1, 2]:
                    method_nodes = self._execute_method(node, method_id)
                    self._add_edges(sub_edges, node, method_nodes)

        # Edge post-processing
        return self._finalize_edges(sub_edges, edge_index.to(device))

    def _execute_method(self, node, action):
        """Execute subgraph method based on action (0: RW, 1: k-hop, 2: k-ego)"""
        if action == 0:
            return self._random_walk(node)
        elif action == 1:
            return self._k_hop_neighbors(node)
        elif action == 2:
            return self._k_ego_neighbors(node)
        return []

    def _add_edges(self, sub_edges, node, nodes):
        """Batch-add bidirectional edges using tensor operations"""
        if not nodes:
            return

        # Convert to tensors on existing device
        node_tensor = torch.tensor([node], device=self.adj_matrix.device)
        nodes_tensor = torch.tensor(nodes, device=self.adj_matrix.device)

        # Create bidirectional edges [src, dst] and [dst, src]
        src = torch.cat([node_tensor.expand(len(nodes)), nodes_tensor])
        dst = torch.cat([nodes_tensor, node_tensor.expand(len(nodes))])

        # Filter invalid nodes (-1 padding)
        valid_mask = (dst >= 0) & (src >= 0)
        edges = torch.stack([src[valid_mask], dst[valid_mask]], dim=0)

        # Extend sub_edges list in-place
        sub_edges.append(edges)

    def _finalize_edges(self, sub_edges, original_edges):
        """Merge and deduplicate edges entirely on GPU"""
        # Convert list of edge tensors to single tensor
        if not sub_edges:
            return original_edges

        device = original_edges.device
        sub_tensors = [e.to(device) for e in sub_edges if e.shape[0] == 2]

        if not sub_tensors:
            return original_edges

        combined = torch.cat([original_edges] + sub_tensors, dim=1)
        return combined.unique(dim=1)  # Deduplicate on GPU

    def _random_walk(self, start_node):
        """Batched random walk using adjacency matrix"""
        walk = torch.full((self.k_rw,), -1,
                        dtype=torch.long, device=self.adj_matrix.device)
        current = torch.as_tensor(start_node, device=self.adj_matrix.device, dtype=torch.long).clone().detach()

        for step in range(self.k_rw):
            neighbors = self.adj_matrix[current]
            valid_neighbors = neighbors[neighbors >= 0]
            if len(valid_neighbors) == 0:
                break
            current = valid_neighbors[torch.randint(0, len(valid_neighbors), (1,))]
            walk[step] = current

        return walk[walk >= 0].tolist()  # Exclude invalid steps

    def _k_hop_neighbors(self, start_node):
        """Vectorized k-hop neighbor finding using adjacency matrix"""
        current_nodes = torch.tensor([start_node], device=self.adj_matrix.device)
        visited = torch.zeros(self.adj_matrix.size(0), dtype=torch.bool,
                            device=self.adj_matrix.device)
        visited[start_node] = True

        for _ in range(self.k_hop):
            # Get all neighbors of current nodes
            neighbors = self.adj_matrix[current_nodes]  # Shape: [batch, max_neighbors]

            # Flatten and remove invalid entries (-1)
            neighbors = neighbors.flatten()
            neighbors = neighbors[neighbors >= 0]

            # Deduplicate and mark visited
            new_nodes = neighbors[~visited[neighbors]]
            visited[new_nodes] = True
            current_nodes = new_nodes

        return visited.nonzero().flatten().tolist()

    def _k_ego_neighbors(self, start_node):
        """BFS using adjacency matrix tensor"""
        visited = torch.zeros(self.adj_matrix.size(0), dtype=torch.bool,
                            device=self.adj_matrix.device)
        queue = torch.tensor([start_node], device=self.adj_matrix.device)
        visited[start_node] = True

        for _ in range(self.k_ego):
            # Get all neighbors of current frontier
            neighbors = self.adj_matrix[queue].flatten()
            neighbors = neighbors[neighbors >= 0]  # Remove padding

            # Filter unvisited
            new_nodes = neighbors[~visited[neighbors]]
            visited[new_nodes] = True
            queue = new_nodes

        return visited.nonzero().flatten().tolist()
# ----------------------------------------------------------------------------------------------------
# 7. GNN Model (Unchanged)
class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden=64, out=2, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ----------------------------------------------------------------------------------------------------
# 8. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

# ----------------------------------------------------------------------------------------------------
# 9. Modified Training Pipeline
def main():
    start_time = time.time()
    # 1. Data Loading & Preprocessing
    df = load_data(sample_frac=0.01)
    if df.empty:
        print("No data loaded - check file paths")
        return

    # Stratified split on raw data (before preprocessing)
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=0.4,
        stratify=df['isFraud'],
        random_state=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_idx_tensor = torch.tensor(train_idx, device=device)
    global_to_local = torch.full((df.shape[0],), -1, dtype=torch.long, device=device)
    global_to_local[train_idx_tensor] = torch.arange(len(train_idx), device=device)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df.loc[temp_idx, 'isFraud'],
        random_state=42
    )

    # Split data chronologically (e.g., by TransactionDT)
    # df = df.sort_values('TransactionDT')
    # train_idx = df.index[:int(0.6 * len(df))]
    # val_idx = df.index[int(0.6 * len(df)) : int(0.8 * len(df))]
    # test_idx = df.index[int(0.8 * len(df)):]

    # Preprocess TRAINING data (fit scalers/PCA)
    train_features, scaler, pca = preprocess_features(
        df.loc[train_idx],
        fit_mode=True
    )

    # Preprocess VAL/TEST data (use training-fitted scalers/PCA)
    val_features = preprocess_features(
        df.loc[val_idx],
        scaler=scaler,
        pca=pca,
        fit_mode=False
    )
    test_features = preprocess_features(
        df.loc[test_idx],
        scaler=scaler,
        pca=pca,
        fit_mode=False
    )

    # Combine features into full tensor (preserve original indices)
    features = torch.zeros((len(df), train_features.shape[1]), dtype=torch.float32)
    features[train_idx] = train_features
    features[val_idx] = val_features
    features[test_idx] = test_features

    features = features.to(device)
    if features.shape[0] == 0:
        print("Feature engineering failed")
        return

    ## 2. FAISS-based Edge Construction
    train_features = features[train_idx]  # Shape: [num_train_nodes, feature_dim]
    original_indices = torch.tensor(train_idx, dtype=torch.long).to(device)  # Maps subset idx → original idx

    # Build edges ONLY within training subgraph
    edge_index = build_semantic_similarity_edges(
        features=train_features,
        original_indices=original_indices,
        threshold=0.8
    )

    # 3. Adaptive MCD Initialization
    print("Started Adaptive MCD")
    train_labels = torch.tensor(df.loc[train_idx, 'isFraud'].values, dtype=torch.long).to(device)
    fraud_nodes_train_local = torch.where(train_labels == 1)[0]
    majority_nodes_train_local = torch.where(train_labels == 0)[0]
    fraud_ratio_train = len(fraud_nodes_train_local) / len(train_idx)

    mcd = AdaptiveMCD(
        input_dim=train_features.size(1),
        fraud_ratio=fraud_ratio_train,
        alpha=0.5
    ).to(device)
    optimizer_mcd = torch.optim.Adam(mcd.parameters(), lr=0.001)

    # 4. MCD Training
    for _ in range(10):
        optimizer_mcd.zero_grad()
        scores = mcd.selector(train_features[majority_nodes_train_local]).squeeze()
        loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
        loss.backward()
        optimizer_mcd.step()

    # 5. Downsample Majority Class
    with torch.no_grad():
        kept_indices = mcd(train_features[majority_nodes_train_local])
        kept_majority_local = majority_nodes_train_local[kept_indices]
        kept_majority_global = torch.tensor(train_idx[kept_indices.cpu().numpy()], device=device)

    # 6. RL Subgraph Enhancement
    print("Started MCES")
    rl_agent = RLAgent(train_features.size(1), global_to_local=global_to_local)
    rl_agent.policy.to(device)

    mces = LiteMCES(k_rw=10, k_hop=3, k_ego=2)
    enhanced_edges = mces.enhance_subgraph(
        edge_index,
        fraud_nodes=fraud_nodes_train_local,
        features=train_features,
        rl_agent=rl_agent,
        y_labels=train_labels
    )

    # 7. Merge Subgraphs
    fraud_nodes_global = torch.tensor(train_idx[fraud_nodes_train_local.cpu().numpy()], device=device)
    combined_nodes = torch.cat([fraud_nodes_global, kept_majority_global])

    mask = torch.isin(edge_index[0], combined_nodes) & \
           torch.isin(edge_index[1], combined_nodes)
    g4_edges = edge_index[:, mask]
    merged_edges = torch.cat([enhanced_edges, g4_edges], dim=1).unique(dim=1)

    # 8. Data Splits & Masks (your existing implementation)
    num_nodes = features.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Create Data object with merged edges
    data = Data(
        x=features,
        edge_index=merged_edges,
        y=torch.tensor(df['isFraud'].values, dtype=torch.long).to(device),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    ).to(device)

    # 9. Class-Weighted Training
    train_fraud_count = len(fraud_nodes_train_local)  # Local training fraud nodes
    train_non_fraud_count = len(kept_majority_local)
    fraud_ratio_final = train_fraud_count / (train_fraud_count + train_non_fraud_count)
    class_weights = torch.tensor([
        1/(1 - fraud_ratio_final),  # Weight for class 0 (downsampled majority)
        1/fraud_ratio_final         # Weight for class 1 (training fraud)
    ], dtype=torch.float32).to(device)

    # Training with class weights
    model = FraudGNN(features.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2)

    # 10. Training with Early Stopping
    best_val_loss = float('inf')
    patience = 30
    counter = 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation Check
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")


    # 11. Final Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()

        # Handle single-class scenario
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            print("\nWarning: Only one class present in predictions")
            print(f"Fraud Cases in Test Set: {(y_true == 1).sum()}/{len(y_true)}")
            print(y_pred)
            end_time = time.time()
            print(f"Training completed in {time.time() - start_time:.2f} seconds.")
            return

        # Safe metric calculation
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Calculate G-Mean safely
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = np.sqrt(tpr * tnr) if (tpr > 0 and tnr > 0) else 0.0

        print(f"\nFinal Test Metrics:")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"G-Means: {gmean:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    end_time = time.time()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
