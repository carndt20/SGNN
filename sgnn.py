import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        
        # Ensure hidden_channels is divisible by num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        
        self.proj = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [num_nodes, hidden_channels]
        batch_size = x.size(0)
        
        # Linear projections
        q = self.query(x)  # [num_nodes, hidden_channels]
        k = self.key(x)    # [num_nodes, hidden_channels]
        v = self.value(x)  # [num_nodes, hidden_channels]
        
        # Reshape for multi-head attention
        head_dim = self.hidden_channels // self.num_heads
        q = q.view(batch_size, self.num_heads, head_dim)  # [num_nodes, num_heads, head_dim]
        k = k.view(batch_size, self.num_heads, head_dim)  # [num_nodes, num_heads, head_dim]
        v = v.view(batch_size, self.num_heads, head_dim)  # [num_nodes, num_heads, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [num_nodes, num_heads, head_dim]
        out = out.view(batch_size, -1)  # [num_nodes, hidden_channels]
        
        # Final projection
        out = self.proj(out)
        return out

class SGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_heads=4):
        super(SGNN, self).__init__()
        
        # Ensure hidden_channels is divisible by num_heads
        hidden_channels = (hidden_channels // num_heads) * num_heads
        
        # Transaction network (GAT layers)
        self.trans_conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, concat=True)
        self.trans_bn1 = nn.BatchNorm1d(hidden_channels)
        self.trans_conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=1, concat=False)
        self.trans_bn2 = nn.BatchNorm1d(hidden_channels // num_heads)
        
        # Identity network (GCN layers for structural features)
        self.ident_conv1 = GCNConv(in_channels, hidden_channels // num_heads)
        self.ident_bn1 = nn.BatchNorm1d(hidden_channels // num_heads)
        self.ident_conv2 = GCNConv(hidden_channels // num_heads, hidden_channels // num_heads)
        self.ident_bn2 = nn.BatchNorm1d(hidden_channels // num_heads)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels // num_heads * 2, hidden_channels // num_heads),
            nn.BatchNorm1d(hidden_channels // num_heads),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Attention fusion
        self.attention = MultiHeadAttention(hidden_channels // num_heads, num_heads=num_heads)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // num_heads, hidden_channels // num_heads),
            nn.BatchNorm1d(hidden_channels // num_heads),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // num_heads, out_channels)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (GCNConv, GATConv)):
                if hasattr(m, 'weight'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, data):
        # Transaction network processing
        trans_x = self.trans_conv1(data.x, data.edge_index)
        trans_x = self.trans_bn1(trans_x)
        trans_x = F.elu(trans_x)
        trans_x = F.dropout(trans_x, p=0.3, training=self.training)
        
        trans_x = self.trans_conv2(trans_x, data.edge_index)
        trans_x = self.trans_bn2(trans_x)
        trans_x = F.elu(trans_x)
        
        # Identity network processing
        ident_x = self.ident_conv1(data.x, data.edge_index)
        ident_x = self.ident_bn1(ident_x)
        ident_x = F.elu(ident_x)
        ident_x = F.dropout(ident_x, p=0.3, training=self.training)
        
        ident_x = self.ident_conv2(ident_x, data.edge_index)
        ident_x = self.ident_bn2(ident_x)
        ident_x = F.elu(ident_x)
        
        # Combine features
        combined_x = torch.cat([trans_x, ident_x], dim=1)
        
        # Fuse features
        fused_x = self.fusion(combined_x)
        
        # Apply self-attention
        attn_x = self.attention(fused_x)
        
        # Final classification
        x = self.classifier(attn_x)
        
        # Stable log softmax with temperature scaling
        temperature = 0.5  # Reduced temperature for sharper predictions
        x = x / temperature
        return F.log_softmax(x, dim=1)

class GraphBuilder:
    def __init__(self, num_nodes=10, num_features=64):
        self.num_nodes = num_nodes
        self.num_features = num_features
        
    def build_transaction_graph(self, transactions=None):
        # Generate random features for nodes
        x = torch.randn((self.num_nodes, self.num_features))
        
        # Generate random edges (sparse connectivity)
        num_edges = self.num_nodes * 2
        edge_index = torch.randint(0, self.num_nodes, (2, num_edges))
        
        # Generate random labels (0 or 1 for binary classification)
        y = torch.randint(0, 2, (self.num_nodes,))
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def build_identity_graph(self, accounts=None):
        # Generate random features for nodes
        x = torch.randn((self.num_nodes, self.num_features))
        
        # Generate random hyperedges (groups of nodes)
        num_hyperedges = self.num_nodes // 2
        hyperedge_size = 3  # Each hyperedge connects 3 nodes
        hyperedge_index = []
        
        for _ in range(num_hyperedges):
            nodes = torch.randint(0, self.num_nodes, (hyperedge_size,))
            hyperedge_index.append(nodes)
        
        hyperedge_index = torch.stack(hyperedge_index).t()
        
        return Data(x=x, hyperedge_index=hyperedge_index) 