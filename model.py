import pandas as pd
import numpy as np
import torch
import faiss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import matplotlib.pyplot as plt  # Added import
import networkx as nx  # Added import


DATA_PATH = r"/content/drive/MyDrive/ieee-fraud-detection"

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
            os.path.join(DATA_PATH, 'train_transaction.csv'),
            dtype=dtypes,
            usecols=list(dtypes.keys()))

        fraud = transactions[transactions['isFraud'] == 1].sample(frac=sample_frac)
        non_fraud_count = int(len(fraud)/0.035*0.965)
        non_fraud = transactions[transactions['isFraud'] == 0].sample(n=non_fraud_count)

        return pd.concat([fraud, non_fraud]).sample(frac=1).reset_index(drop=True)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------------------------------------
# 2. Feature Engineering (Unchanged)
def preprocess_features(df):
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    df['card4'] = df['card4'].fillna('unknown')

    # Convert TransactionDT to hourly bins
    df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24
    
    # Time since last transaction (global)
    df = df.sort_values('TransactionDT')
    df['TimeSinceLast'] = df['TransactionDT'].diff().fillna(0)
    
    # 3. Numerical features (now includes temporal features)
    num_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 
                'TransactionHour', 'TimeSinceLast']
    num_features = df[num_cols].fillna(0)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features)

    email_hash = pd.get_dummies(pd.util.hash_pandas_object(df['P_emaildomain']) % 50, prefix='email')
    card_hash = pd.get_dummies(pd.util.hash_pandas_object(df['card4']) % 20, prefix='card')

    features = np.hstack([num_scaled, email_hash, card_hash])

    n_components = min(32, features.shape[1])
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(features), dtype=torch.float32)

# ----------------------------------------------------------------------------------------------------
# 3. New: Semantic Similarity Edge Construction (Equation 1)
def build_semantic_similarity_edges(features, threshold=0.8, batch_size=8192):
    """FAISS-accelerated edge construction"""
    edge_index = []
    num_nodes = features.size(0)
    features_np = features.numpy()
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(features_np)
    index = faiss.IndexFlatIP(features_np.shape[1])
    
    print("FAISS indexing started")
    index.add(features_np)
    
    for i in range(0, num_nodes, batch_size):
        batch = features_np[i:i+batch_size]
        similarities, neighbors = index.search(batch, 100)  # Top 100 neighbors
        
        for idx_in_batch, (sim_row, nbr_row) in enumerate(zip(similarities, neighbors)):
            src = i + idx_in_batch
            valid = sim_row > threshold
            for dst in nbr_row[valid]:
                if src != dst:
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])

    print(f"Generated {len(edge_index)//2} edges (FAISS)")
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_tensor.unique(dim=1)

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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.005)
        self.mces = LiteMCES()
        
    def train_rl(self, nodes, features, edge_index, y_labels, n_epochs=100):
        # Train on 20% subset
        subset = nodes[torch.randperm(len(nodes))[:int(0.2*len(nodes))]]
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            losses = []
            for node in subset:
                probs, logits = self.policy(features[node])
                action = torch.multinomial(probs, 1).item()
                
                # Get subgraph using chosen method
                nodes = self.mces._execute_method(edge_index, node, action)
                reward = len([n for n in nodes if y_labels[n] == 1])
                
                # Policy gradient update
                loss = -torch.log(probs[action]) * reward
                losses.append(loss)
                epoch_reward += reward
            
            # Update model
            self.optimizer.zero_grad()
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_reward = epoch_reward / len(subset)
                print(f"RL Epoch {epoch+1}/{n_epochs} | "
                      f"Avg Loss: {epoch_loss:.2f} | "
                      f"Avg Reward: {avg_reward:.2f}")

# ----------------------------------------------------------------------------------------------------
# 6. Minor-node-centered explored Subgraph (MCES)
class LiteMCES:
    def __init__(self, k_rw=10, k_hop=3, k_ego=2):
        self.k_rw = k_rw  # Random walk length (paper: 10)
        self.k_hop = k_hop  # K-hop neighbors (paper: 3)
        self.k_ego = k_ego  # K-ego depth (paper: 2)
        self.max_neighbors = 1000  # Truncation limit

    def enhance_subgraph(self, edge_index, fraud_nodes, features, rl_agent=None, y_labels=None):
        sub_edges = []
        
        if rl_agent:  # RL-enhanced path
            # Training handled internally by RLAgent
            rl_agent.train_rl(fraud_nodes, features, edge_index, y_labels)
            
            # Inference using trained policy
            for node in fraud_nodes:
                with torch.no_grad():
                    action_probs = rl_agent.policy(features[node])[0]
                    action = action_probs.argmax().item()

                # Single method execution based on RL choice
                method_nodes = self._execute_method(edge_index, node, action)
                self._add_edges(sub_edges, node, method_nodes)
                
        else:  # Original three-method path
            for node in fraud_nodes:
                # Execute all three methods
                for method_id in [0, 1, 2]:
                    method_nodes = self._execute_method(edge_index, node, method_id)
                    self._add_edges(sub_edges, node, method_nodes)

        # Edge post-processing
        return self._finalize_edges(sub_edges, edge_index)

    def _execute_method(self, edge_index, node, action):
        """Unified method execution"""
        node = node.item()  # Handle tensor conversion
        if action == 0:
            return self._random_walk(edge_index, node)
        elif action == 1:
            return self._k_hop_neighbors(edge_index, node)
        elif action == 2:
            return self._k_ego_neighbors(edge_index, node)
        return []

    def _add_edges(self, sub_edges, node, nodes):
        """Bidirectional edge addition"""
        for n in nodes:
            sub_edges.append([node.item(), n])
            sub_edges.append([n, node.item()])

    def _finalize_edges(self, sub_edges, original_edges):
        """Deduplication and merging"""
        if not sub_edges:
            return original_edges
        
        edge_tensor = torch.tensor(sub_edges, dtype=torch.long).t()
        unique_edges = edge_tensor.unique(dim=1)
        
        # Merge with original semantic edges
        return torch.cat([original_edges, unique_edges], dim=1).unique(dim=1)

    def _random_walk(self, edge_index, start_node):
        walk = [start_node]
        current = start_node
        for _ in range(self.k_rw):
            # Get neighbors and truncate to 500
            neighbors = edge_index[1][edge_index[0] == current].tolist()[:self.max_neighbors]
            if not neighbors:
                break
            current = np.random.choice(neighbors)
            walk.append(current)
        return walk[1:]  # Exclude self-loop

    def _k_hop_neighbors(self, edge_index, start_node):
        neighbors = set()
        current = {start_node}
        for _ in range(self.k_hop):
            next_neighbors = set()
            for n in current:
                # Get neighbors and truncate to 500
                nbrs = edge_index[1][edge_index[0] == n].tolist()[:self.max_neighbors]
                next_neighbors.update(nbrs)
            # Truncate to 500 nodes for next hop
            next_neighbors = set(list(next_neighbors)[:self.max_neighbors])
            neighbors.update(next_neighbors)
            current = next_neighbors - neighbors
        return neighbors - {start_node}

    def _k_ego_neighbors(self, edge_index, start_node):
      visited = set()
      current_layer = {start_node}
      visited.add(start_node)  # Track all visited nodes

      # Precompute adjacency list for faster lookups
      adj_list = defaultdict(list)
      for src, dst in edge_index.t().tolist():
          adj_list[src].append(dst)

      for _ in range(self.k_ego):
          next_layer = []
          # Process nodes in batches for better cache utilization
          batch_nodes = list(current_layer)

          for node in batch_nodes:
              # Get neighbors not already visited (using precomputed adj list)
              neighbors = [n for n in adj_list.get(node, []) if n not in visited]

              # Add up to remaining capacity (500 total per layer)
              remaining = self.max_neighbors - len(next_layer)
              if remaining <= 0:
                  break
              next_layer.extend(neighbors[:remaining])

          # Deduplicate and truncate exactly to 500
          next_layer = list(set(next_layer))[:self.max_neighbors]

          # Update tracking structures
          visited.update(next_layer)
          current_layer = set(next_layer)

      return visited - {start_node}

# ----------------------------------------------------------------------------------------------------
# 7. GNN Model (Unchanged)
class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden=128, out=2, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

# ----------------------------------------------------------------------------------------------------
# 8. Modified Training Pipeline
def main():
  # 1. Data Loading & Preprocessing
    df = load_data(sample_frac=0.01)
    if df.empty:
        print("No data loaded - check file paths")
        return
                         
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = preprocess_features(df)
    features = features.to(device)
    if features.shape[0] == 0:
        print("Feature engineering failed")
        return

    ## 2. FAISS-based Edge Construction
    edge_index = build_semantic_similarity_edges(features, threshold=0.8)

    # 3. Adaptive MCD Initialization
    print("Started Adaptive MCD")
    fraud_nodes = torch.where(torch.tensor(df['isFraud'].values) == 1)[0]
    majority_nodes = torch.where(torch.tensor(df['isFraud'].values) == 0)[0]
    fraud_ratio = len(fraud_nodes) / len(df)
    
    mcd = AdaptiveMCD(
        input_dim=features.size(1),
        fraud_ratio=fraud_ratio,
        alpha=1.2  # Tunable hyperparameter
    )
    optimizer_mcd = torch.optim.Adam(mcd.parameters(), lr=0.01)

    # 4. MCD Training
    for _ in range(10):  # Paper uses 10 epochs
        optimizer_mcd.zero_grad()
        scores = mcd.selector(features[majority_nodes]).squeeze()
        loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
        loss.backward()
        optimizer_mcd.step()

    # 5. Downsample Majority Class
    with torch.no_grad():
        kept_indices = mcd(features[majority_nodes])
        kept_majority = majority_nodes[kept_indices]

    #  print(f"Kept {len(kept_majority)}/{len(majority_nodes)} majority nodes") !Remove!

    # 6. RL Subgraph Enhancement
    print("Started MCES")
    rl_agent = RLAgent(features.size(1))
    rl_agent.policy.to(device)
    mces = LiteMCES(k_rw=10, k_hop=3, k_ego=2)
    enhanced_edges = mces.enhance_subgraph(
        edge_index, 
        fraud_nodes,
        features,
        rl_agent=rl_agent,  # RL-enhanced generation
        y_labels=torch.tensor(df['isFraud'].values, dtype=torch.long).to(device)  # Pass actual labels here
    )

    # 7. Merge Subgraphs
    mask = torch.isin(edge_index[0], torch.cat([fraud_nodes, kept_majority])) & \
           torch.isin(edge_index[1], torch.cat([fraud_nodes, kept_majority]))
    g4_edges = edge_index[:, mask]
    merged_edges = torch.cat([enhanced_edges, g4_edges], dim=1).unique(dim=1)
    merged_edges = merged_edges.to(device)

    # 8. Data Splits & Masks (your existing implementation)
    final_nodes = torch.cat([fraud_nodes, kept_majority])

    print("Final Nodes: ", len(final_nodes))
    print("Kept Majority: ", len(kept_majority))
    print("Fraud Nodes: ", len(fraud_nodes))

    # Create proper boolean masks
    train_idx, temp_idx = train_test_split(
        final_nodes.cpu().numpy(),
        test_size=0.4,
        stratify=df['isFraud'].iloc[final_nodes]
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df['isFraud'].iloc[temp_idx]
    )

    num_nodes = features.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Create Data object with merged edges
    data = Data(
        x=features,
        edge_index=merged_edges,
        y=torch.tensor(df['isFraud'].values, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data = data.to(device)

    # 9. Class-Weighted Training
    fraud_ratio_final = len(fraud_nodes)/len(final_nodes)
    class_weights = torch.tensor([
        1/(1 - fraud_ratio_final), 
        1/fraud_ratio_final
    ], dtype=torch.float32).cpu()

    # Training with class weights
    model = FraudGNN(features.size(1))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.NLLLoss(weight=class_weights)

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
            return

        # Safe metric calculation
        recall = recall = recall_score(y_true, y_pred)
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

# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()