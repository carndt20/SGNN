import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

class EllipticLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.features = None
        self.edges = None
        self.labels = None
        self.node_mapping = None
        
    def load_data(self):
        # Load features
        print("Loading features...")
        features_df = pd.read_csv(f'{self.data_dir}/txs_features.csv')
        
        # Load edges
        print("Loading edges...")
        edges_df = pd.read_csv(f'{self.data_dir}/txs_edgelist.csv')
        
        # Load labels
        print("Loading labels...")
        labels_df = pd.read_csv(f'{self.data_dir}/txs_classes.csv')
        
        # Print unique classes for debugging
        print("Unique classes in dataset:", labels_df['class'].unique())
        
        # Create node mapping
        unique_nodes = features_df['txId'].unique()
        self.node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Convert features to tensor and handle missing values
        feature_cols = [col for col in features_df.columns if col != 'txId']
        features = features_df[feature_cols].values
        
        # Handle missing values (replace with zeros)
        features = np.nan_to_num(features, nan=0.0)
        
        # Convert boolean values to binary
        features = features.astype(np.float32)
        
        # Normalize features using StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.features = torch.FloatTensor(features)
        
        # Convert edges to tensor and ensure they're directed
        edge_index = torch.LongTensor([
            [self.node_mapping[src] for src in edges_df['txId1']],
            [self.node_mapping[dst] for dst in edges_df['txId2']]
        ])
        
        # Add reverse edges to make the graph undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Convert labels to tensor (1: illicit, 0: licit, -1: unknown)
        # Map classes: 1 -> illicit (1), 2 -> licit (0), 3 -> licit (0), unknown -> unknown (-1)
        labels_dict = {'unknown': -1, '1': 1, '2': 0, '3': 0, 1: 1, 2: 0, 3: 0}
        labels = []
        for node in unique_nodes:
            label_row = labels_df[labels_df['txId'] == node]
            if len(label_row) > 0:
                label_val = label_row['class'].iloc[0]
                labels.append(labels_dict[label_val])
            else:
                labels.append(-1)  # Unknown label
        
        self.labels = torch.LongTensor(labels)
        
        # Print label distribution for debugging
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("Label distribution:", dict(zip(unique_labels, counts)))
        
        # Create PyTorch Geometric Data object
        data = Data(x=self.features, edge_index=edge_index, y=self.labels)
        
        return data
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        data = self.load_data()
        
        # Get indices of known labels (not -1)
        known_indices = (data.y != -1).nonzero().squeeze()
        
        # Split known indices into train and test with stratification
        train_idx, test_idx = train_test_split(
            known_indices.numpy(),
            test_size=test_size,
            random_state=random_state,
            stratify=data.y[known_indices].numpy()
        )
        
        # Convert back to tensors
        train_idx = torch.LongTensor(train_idx)
        test_idx = torch.LongTensor(test_idx)
        
        # Create masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.test_mask = test_mask
        
        # Print class distribution in train and test sets
        train_labels = data.y[train_mask].cpu().numpy()
        test_labels = data.y[test_mask].cpu().numpy()
        print(f'Training class distribution: {np.bincount(train_labels)}')
        print(f'Testing class distribution: {np.bincount(test_labels)}')
        
        return data 