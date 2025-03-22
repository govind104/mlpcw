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
def load_data(sample_frac=0.3, random_state=42):
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
        # Load full dataset
        transactions = pd.read_csv(
            'train_transaction.csv',
            dtype=dtypes,
            usecols=list(dtypes.keys())
        )

        # Sort by TransactionDT to ensure chronological order
        transactions = transactions.sort_values('TransactionDT').reset_index(drop=True)

        # Fixed Proportion Split (Train: 60%, Val: 20%, Test: 20%)
        train_end = int(0.6 * len(transactions))  # First 60% for training
        val_end = int(0.8 * len(transactions))    # Next 20% for validation
        
        train_df = transactions.iloc[:train_end]  # First 60%
        val_df = transactions.iloc[train_end:val_end]  # Next 20%
        test_df = transactions.iloc[val_end:]  # Last 20%

        # Balance ONLY the training set
        fraud_train = train_df[train_df['isFraud'] == 1].sample(frac=sample_frac, random_state=random_state)
        non_fraud_count = int(len(fraud_train) / 0.035 * 0.965)  # Maintain original ratio
        non_fraud_train = train_df[train_df['isFraud'] == 0].sample(n=non_fraud_count, random_state=random_state)
        
        # Combine and shuffle balanced training data
        balanced_train = pd.concat([fraud_train, non_fraud_train]).sample(frac=1, random_state=random_state)
        
        return balanced_train.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    except Exception as e:
        print(f"Data loading failed: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------------------------------------
# 2. Feature Engineering (Unchanged)
def preprocess_features(train_df, val_df=None, test_df=None):
    """
    Preprocess features for train, validation, and test sets.
    Fits scalers and PCA on the training set and applies them to validation/test sets.
    """
    def _process_single_df(df, scaler=None, pca=None, is_train=False):
        """Helper function to process a single dataframe."""
        # Fill missing values
        df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
        df['card4'] = df['card4'].fillna('unknown')

        # Convert TransactionDT to hourly bins
        df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24

        # Time since last transaction (global)
        df = df.sort_values('TransactionDT')
        df['TimeSinceLast'] = df['TransactionDT'].diff().fillna(0)

        # Numerical features (now includes temporal features)
        num_cols = ['TransactionAmt', 'C1', 'C2', 'C3',
                    'TransactionHour', 'TimeSinceLast']
        num_features = df[num_cols].fillna(0)

        # Fit scaler and PCA on training data, transform on validation/test data
        if is_train:
            scaler = StandardScaler().fit(num_features)
            num_scaled = scaler.transform(num_features)
            pca = PCA(n_components=min(32, num_scaled.shape[1])).fit(num_scaled)
        else:
            num_scaled = scaler.transform(num_features)

        # Apply PCA
        pca_features = pca.transform(num_scaled)

        # Hash categorical features
        email_hash = pd.get_dummies(pd.util.hash_pandas_object(df['P_emaildomain']) % 50, prefix='email')
        card_hash = pd.get_dummies(pd.util.hash_pandas_object(df['card4']) % 20, prefix='card')

        # Combine numerical and categorical features
        features = np.hstack([pca_features, email_hash, card_hash])
        return torch.tensor(features, dtype=torch.float32), scaler, pca

    # Process training data (fit scaler and PCA)
    features_train, scaler, pca = _process_single_df(train_df, is_train=True)

    # Process validation data (use fitted scaler and PCA)
    if val_df is not None:
        features_val, _, _ = _process_single_df(val_df, scaler=scaler, pca=pca)
    else:
        features_val = None

    # Process test data (use fitted scaler and PCA)
    if test_df is not None:
        features_test, _, _ = _process_single_df(test_df, scaler=scaler, pca=pca)
    else:
        features_test = None

    return features_train, features_val, features_test, scaler, pca

# ----------------------------------------------------------------------------------------------------
# 3. New: Semantic Similarity Edge Construction (Equation 1)
def build_semantic_similarity_edges(features, threshold=0.8, split: str = None, batch_size=8192):
    """FAISS-accelerated edge construction"""
    edge_index = []
    num_nodes = features.size(0)
    features_np = features.cpu().numpy()
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(features_np)
    index = faiss.IndexFlatIP(features_np.shape[1])  # Inner product for cosine similarity

    print(f"FAISS indexing started for {num_nodes} {split} nodes")
    index.add(features_np)  # Build index on the input features

    for i in range(0, num_nodes, batch_size):
        batch = features_np[i:i + batch_size]
        similarities, neighbors = index.search(batch, 100)  # Top 100 neighbors

        for idx_in_batch, (sim_row, nbr_row) in enumerate(zip(similarities, neighbors)):
            src = i + idx_in_batch  # Source node index
            valid = sim_row > threshold  # Filter by similarity threshold
            for dst in nbr_row[valid]:
                if src != dst:  # Avoid self-loops
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Add bidirectional edges

    print(f"Generated {len(edge_index) // 2} edges (FAISS)")
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long).to(features.device)

    # Convert to tensor and deduplicate edges
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t()
    edge_tensor = edge_tensor.unique(dim=1)  # Deduplicate edges
    return edge_tensor.to(features.device)  # Move to the same device as features

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
    def __init__(self, feat_dim):
        self.policy = SubgraphPolicy(feat_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.mces = LiteMCES()

    def train_rl(self, nodes, features, edge_index, y_labels, n_epochs=50):
        """Batched RL training with average loss tracking"""
        device = features.device
        y_labels = y_labels.to(device)
        subset = nodes[torch.randperm(len(nodes))[:int(0.1 * len(nodes))]].to(device)
        
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
                reward = torch.log1p(y_labels[method_nodes].sum().float())  # Normalized reward
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

            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if epoch >= 10 else loss.item()
                print(f"RL Epoch {epoch + 1}: Loss={avg_loss:.4f}, Avg Reward={rewards.mean().item():.4f}")

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
        """Convert edge_index to dense adjacency list on GPU."""
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
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ----------------------------------------------------------------------------------------------------
# 8. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ----------------------------------------------------------------------------------------------------
# 9. Modified Training Pipeline
def main():

    start_time = time.time()
    # 1. Data Loading & Preprocessing
    train_df, val_df, test_df = load_data(sample_frac=0.05)
    if train_df.empty or val_df.empty or test_df.empty:
        print("Data loaded and split incorrectly, please check.")
        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Validation DataFrame shape: {val_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")
        return

    # device = torch.device('cpu')
    features_train, features_val, features_test, scaler, pca = preprocess_features(train_df, val_df, test_df)
    if features_train.shape[0] == 0 or features_val.shape[0] == 0 or features_test.shape[0] == 0:
        print("Feature engineering failed")
        print(f"Train features shape: {features_train.shape}")
        print(f"Validation features shape: {features_val.shape}")
        print(f"Test features shape: {features_test.shape}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_train = features_train.to(device)
    features_val = features_val.to(device)
    features_test = features_test.to(device)

    ## 2. FAISS-based Edge Construction
    edge_index_train = build_semantic_similarity_edges(features_train, threshold=0.8, split='train')
    edge_index_val = build_semantic_similarity_edges(features_val, threshold=0.8, split='validation')
    edge_index_test = build_semantic_similarity_edges(features_test, threshold=0.8, split='test')

    # 3. Adaptive MCD Initialization
    print("Started Adaptive MCD")
    train_labels = torch.tensor(train_df['isFraud'].values, dtype=torch.long).to(device)
    fraud_nodes_train = torch.where(train_labels == 1)[0]
    majority_nodes_train = torch.where(train_labels == 0)[0]    
    fraud_ratio = len(fraud_nodes_train) / len(train_labels)

    mcd = AdaptiveMCD(
        input_dim=features_train.size(1),
        fraud_ratio=fraud_ratio,
        alpha=0.5  # Tunable hyperparameter
    ).to(device)

    # 4. MCD Training
    optimizer_mcd = torch.optim.Adam(mcd.parameters(), lr=0.001)
    for _ in range(10):  # Paper uses 10 epochs
        optimizer_mcd.zero_grad()
        scores = mcd.selector(features_train[majority_nodes_train]).squeeze()
        loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
        loss.backward()
        optimizer_mcd.step()

    # 5. Downsample Majority Class
    with torch.no_grad():
        kept_indices = mcd(features_train[majority_nodes_train])
        kept_majority = majority_nodes_train[kept_indices]

    print(f"Kept {len(kept_majority)}/{len(majority_nodes_train)} majority nodes")
    final_train_nodes = torch.cat([fraud_nodes_train, kept_majority])

    # 6. RL Subgraph Enhancement
    print("Started MCES")
    rl_agent = RLAgent(features_train.size(1))
    rl_agent.policy.to(device)
    mces = LiteMCES(k_rw=10, k_hop=3, k_ego=2)
    enhanced_edges = mces.enhance_subgraph(
        edge_index_train,
        fraud_nodes_train,
        features_train,
        rl_agent=rl_agent,  # RL-enhanced generation
        y_labels=train_labels  # Pass actual labels here
    )

    # 7. Merge Subgraphs
    merged_edges = torch.cat([edge_index_train, enhanced_edges], dim=1).unique(dim=1)
    merged_edges = merged_edges.to(device)

    # 8. Data Splits & Masks
    print("Final Nodes: ", len(final_train_nodes))
    print("Kept Majority: ", len(kept_majority))
    print("Fraud Nodes: ", len(fraud_nodes_train))

    train_mask = torch.zeros(features_train.size(0), dtype=torch.bool, device=device)
    val_mask = torch.zeros(features_val.size(0), dtype=torch.bool, device=device)
    test_mask = torch.zeros(features_test.size(0), dtype=torch.bool, device=device)

    train_mask[final_train_nodes] = True  # Final training nodes (fraud + downsampled majority)
    val_mask[:] = True  # All validation nodes are used for validation
    test_mask[:] = True  # All test nodes are used for testing

    # Create Data object for training
    train_data = Data(
        x=features_train,
        edge_index=merged_edges,
        y=train_labels,
        train_mask=train_mask,
        val_mask=torch.zeros(features_train.size(0), dtype=torch.bool, device=device),
        test_mask=torch.zeros(features_train.size(0), dtype=torch.bool, device=device)
    ).to(device)

    # Create Data objects for validation and testing
    val_data = Data(
        x=features_val,
        edge_index=edge_index_val,  # Use original edges for validation
        y=torch.tensor(val_df['isFraud'].values, dtype=torch.long).to(device),
        train_mask=torch.zeros(features_val.size(0), dtype=torch.bool, device=device),
        val_mask=val_mask,
        test_mask=torch.zeros(features_val.size(0), dtype=torch.bool, device=device)
    ).to(device)

    test_data = Data(
        x=features_test,
        edge_index=edge_index_test,  # Use original edges for testing
        y=torch.tensor(test_df['isFraud'].values, dtype=torch.long).to(device),
        train_mask=torch.zeros(features_test.size(0), dtype=torch.bool, device=device),
        val_mask=torch.zeros(features_test.size(0), dtype=torch.bool, device=device),
        test_mask=test_mask
    ).to(device)

    # 9. Class-Weighted Training
    fraud_ratio_final = len(fraud_nodes_train) / len(final_train_nodes)
    class_weights = torch.tensor([
        1 / (1 - fraud_ratio_final),  # Weight for non-fraud class
        1 / fraud_ratio_final         # Weight for fraud class
    ], dtype=torch.float32).to(device)

    # Training with class weights
    model = FraudGNN(features_train.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2, weight=class_weights)

    # 10. Training with Early Stopping
    best_val_loss = float('inf')
    patience = 30
    counter = 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation Check
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x, val_data.edge_index)
                val_loss = criterion(val_out[val_data.val_mask], val_data.y[val_data.val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")

    # 11. Final Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_data.x, test_data.edge_index).argmax(dim=1)
        y_true = test_data.y[test_data.test_mask].cpu().numpy()
        y_pred = pred[test_data.test_mask].cpu().numpy()

        # Handle single-class scenario
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            print("\nWarning: Only one class present in predictions")
            print(f"Fraud Cases in Test Set: {(y_true == 1).sum()}/{len(y_true)}")
            print(y_pred)
            return

        # Safe metric calculation
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = np.sqrt(tpr * tnr) if (tpr > 0 and tnr > 0) else 0.0

        print(f"\nFinal Metrics:")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"G-Means: {gmean:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    end_time = time.time()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
