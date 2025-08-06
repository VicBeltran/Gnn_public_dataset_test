import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, EdgeConv, HeteroConv
from torch_geometric.utils import add_self_loops, sort_edge_index
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
    """Shared GraphSAGE model for fraud detection with LSTM mode"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):  # Reduced dropout
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='lstm')  # LSTM aggregation
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')  # LSTM aggregation
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='lstm')  # LSTM aggregation
        self.dropout = dropout
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third layer (final)
        x = self.conv3(x, edge_index)
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get intermediate embeddings (128-dim) for caching"""
        # First layer
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer (this will be our 128-dim embedding)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class EdgeGraphSAGE(torch.nn.Module):
    """GraphSAGE model for edge-level fraud detection"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='lstm')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')
        
        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels + edge_dim, hidden_channels),  # Changed from hidden_channels * 2 + edge_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Edge-level prediction
        # Get source and target node embeddings for each edge
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col], edge_attr], dim=1)
        
        # Predict edge labels
        edge_out = self.edge_mlp(edge_feat)
        return edge_out
    
    def get_embeddings(self, x, edge_index, edge_attr):
        """Get node embeddings for caching"""
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class ImprovedEdgeGraphSAGE(torch.nn.Module):
    """Improved GraphSAGE model for edge-level fraud detection with residual connections"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='lstm')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')
        
        # Edge prediction head with attention
        # Ensure embedding dimension is divisible by number of heads
        attention_dim = hidden_channels * 2
        num_heads = 4
        # Adjust attention_dim to be divisible by num_heads
        if attention_dim % num_heads != 0:
            attention_dim = ((attention_dim // num_heads) + 1) * num_heads
        
        self.edge_attention = nn.MultiheadAttention(attention_dim, num_heads=num_heads, dropout=dropout)
        
        # Add projection layer if needed
        if hidden_channels * 2 != attention_dim:
            self.edge_projection = nn.Linear(hidden_channels * 2, attention_dim)
        else:
            self.edge_projection = None
        
        # Edge prediction head - calculate input dimension dynamically
        # Input will be: hidden_channels (from gated embeddings) + edge_dim (from edge attributes)
        edge_mlp_input_dim = hidden_channels + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr):
        # Node embeddings with residual connections
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(x2 + x1)  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(x3 + x2)  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        x4 = self.conv4(x3, edge_index)
        x4 = self.relu(x4 + x3)  # Residual connection
        x4 = F.dropout(x4, p=self.dropout, training=self.training)
        
        # Edge-level prediction with attention
        row, col = edge_index
        edge_feat = torch.cat([x4[row], x4[col]], dim=1)
        
        # Project edge features if needed
        if self.edge_projection is not None:
            edge_feat = self.edge_projection(edge_feat)
        
        # Apply attention to edge features
        edge_feat_reshaped = edge_feat.unsqueeze(0)  # Add batch dimension
        edge_feat_attended, _ = self.edge_attention(edge_feat_reshaped, edge_feat_reshaped, edge_feat_reshaped)
        edge_feat_attended = edge_feat_attended.squeeze(0)  # Remove batch dimension
        
        # Combine with original edge attributes
        edge_feat_final = torch.cat([edge_feat_attended, edge_attr], dim=1)
        
        # Predict edge labels
        edge_out = self.edge_mlp(edge_feat_final)
        return edge_out
    
    def get_embeddings(self, x, edge_index, edge_attr):
        """Get node embeddings for caching"""
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(x2 + x1)  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(x3 + x2)  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        return x3

class FraudClassifier(nn.Module):
    """Enhanced classifier for fraud detection with reduced dropout"""
    def __init__(self, input_dim=264, hidden_dim=128, output_dim=1, dropout=0.1):  # Updated for edge-level features
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Reduced dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim // 4)
        
    def forward(self, x):
        # Handle single sample inference by using eval mode or skipping batch norm
        if x.size(0) == 1:
            # For single sample, use eval mode for batch norm
            self.batch_norm1.eval()
            self.batch_norm2.eval()
            self.batch_norm3.eval()
        
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

class ImprovedGraphSAGE(torch.nn.Module):
    """Improved GraphSAGE model with LSTM attention and residual connections"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):  # Reduced dropout
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='lstm')  # LSTM aggregation
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')  # LSTM aggregation
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')  # LSTM aggregation
        self.conv4 = SAGEConv(hidden_channels, out_channels, aggr='lstm')  # LSTM aggregation
        self.dropout = dropout
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        # First layer
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second layer with residual connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(x2 + x1)  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Third layer with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(x3 + x2)  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        # Final layer
        x4 = self.conv4(x3, edge_index)
        return x4
    
    def get_embeddings(self, x, edge_index):
        """Get intermediate embeddings (128-dim) for caching"""
        # First layer
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second layer with residual connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(x2 + x1)  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Third layer with residual connection (this will be our 128-dim embedding)
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(x3 + x2)  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        return x3 

class HeterogeneousEdgeGraphSAGE(torch.nn.Module):
    """Hetero GraphSAGE for edge-level fraud with proper PyG support"""
    def __init__(self, source_feature_dim, target_feature_dim,
                 hidden_channels, out_channels, edge_dim, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.relu = nn.ReLU()

        # Encoders for each node type
        self.source_encoder = nn.Sequential(
            nn.Linear(source_feature_dim, hidden_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Build three heterogeneous SAGEConv layers
        def make_layer():
            return HeteroConv({
                ('source','to','source'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
                ('target','to','target'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
                ('source','to','target'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
                ('target','to','source'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
            }, aggr='mean')
        self.conv1 = make_layer()
        self.conv2 = make_layer()
        self.conv3 = make_layer()

        # Edge attention gate
        self.edge_gate = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, 1), nn.Sigmoid()
        )
        # Final MLP for edge prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels + edge_dim, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1) Encode node features
        x_enc = {
            'source': self.source_encoder(x_dict['source']),
            'target': self.target_encoder(x_dict['target'])
        }
        device = x_enc['source'].device
        edge_dim = next(iter(edge_attr_dict.values())).size(1)

        # 2) Prepare edge indices and attributes (no self-loops needed for mean aggregation)
        loops = {}
        attrs = {}
        for et, idx in edge_index_dict.items():
            loops[et] = idx
            attrs[et] = edge_attr_dict.get(et, torch.empty((0, edge_dim), dtype=torch.float, device=device))

        # Ensure all expected edge types present
        for et in [('source','to','source'), ('target','to','target'),
                   ('source','to','target'), ('target','to','source')]:
            loops.setdefault(et, torch.empty((2,0), dtype=torch.long, device=device))
            attrs.setdefault(et, torch.empty((0, edge_dim), dtype=torch.float, device=device))

        # 3) Heterogeneous message passing with residuals
        x1 = self.conv1(x_enc, loops)
        x1 = {k: self.relu(v) for k,v in x1.items()}
        x1 = {k: F.dropout(v, p=self.dropout, training=self.training) for k,v in x1.items()}

        x2 = self.conv2(x1, loops)
        x2 = {k: self.relu(v + x1[k]) for k,v in x2.items()}
        x2 = {k: F.dropout(v, p=self.dropout, training=self.training) for k,v in x2.items()}

        x3 = self.conv3(x2, loops)
        x3 = {k: self.relu(v + x2[k]) for k,v in x3.items()}
        x3 = {k: F.dropout(v, p=self.dropout, training=self.training) for k,v in x3.items()}

        # 4) Edge-level output: prioritize source->target
        for et in [('source','to','target'), ('target','to','source'),
                   ('source','to','source'), ('target','to','target')]:
            idx = loops[et]
            if idx.size(1) > 0:
                row, col = idx
                src_h = x3[et[0]][row]
                tgt_h = x3[et[2]][col]
                gate = self.edge_gate(torch.cat([src_h, tgt_h], dim=1))
                combined = src_h * gate + tgt_h * (1 - gate)
                feat = torch.cat([combined, attrs[et]], dim=1)
                return self.edge_mlp(feat)

        # No edges -> return empty
        return torch.empty((0, self.edge_mlp[-1].out_features), device=device)

    def get_embeddings(self, x_dict, edge_index_dict, edge_attr_dict):
        x_enc = {
            'source': self.source_encoder(x_dict['source']),
            'target': self.target_encoder(x_dict['target'])
        }
        x1 = self.conv1(x_enc, edge_index_dict)
        x1 = {k: self.relu(v) for k,v in x1.items()}
        x1 = {k: F.dropout(v, p=self.dropout, training=self.training) for k,v in x1.items()}
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {k: self.relu(v + x1[k]) for k,v in x2.items()}
        return x2




class HeterogeneousEdgeGraphSAGELSTMv1(torch.nn.Module):
    """Improved heterogeneous GraphSAGE model for edge-level fraud detection with proper PyG support"""
    def __init__(self, source_feature_dim, target_feature_dim, hidden_channels, out_channels, edge_dim, dropout=0.1):
        super().__init__()
        
        # Store original feature dimensions
        self.source_feature_dim = source_feature_dim
        self.target_feature_dim = target_feature_dim
        
        # Separate encoders for different node types (no padding needed)
        self.source_encoder = nn.Sequential(
            nn.Linear(source_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Heterogeneous graph convolution layers - use 'lstm' aggregation for better performance
        self.conv1 = HeteroConv({
            ('source', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('source', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('source', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('source', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
        }, aggr='mean')
        
        self.conv3 = HeteroConv({
            ('source', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('source', 'to', 'target'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
            ('target', 'to', 'source'): SAGEConv(hidden_channels, hidden_channels, aggr='lstm'),
        }, aggr='mean')
        
        # Lightweight edge attention via gating (O(n) instead of O(n²))
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels + edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout
        self.relu = nn.ReLU()
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Forward pass with proper heterogeneous support
        
        Args:
            x_dict: Dictionary of node features per type {'source': tensor, 'target': tensor}
            edge_index_dict: Dictionary of edge indices per type
            edge_attr_dict: Dictionary of edge attributes per type
        """
        # Encode different node types separately
        x_encoded = {
            'source': self.source_encoder(x_dict['source']),
            'target': self.target_encoder(x_dict['target'])
        }
        
        edge_index_dict_with_self_loops = {}
        edge_attr_dict_with_self_loops = {}
        
        # Define all expected edge types
        expected_edge_types = [
            ('source', 'to', 'source'),
            ('target', 'to', 'target'),
            ('source', 'to', 'target'),
            ('target', 'to', 'source')
        ]
        
        # Filter out empty edge types and process non-empty ones
        non_empty_edge_types = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(1) > 0:
                non_empty_edge_types[edge_type] = edge_index
        
        for edge_type, edge_index in non_empty_edge_types.items():
            # Add self-loops to existing edges
            src_type = edge_type[0]
            tgt_type = edge_type[2]
            num_src_nodes = x_encoded[src_type].size(0)
            num_tgt_nodes = x_encoded[tgt_type].size(0)
            
            # Use the target node count for add_self_loops since edges point to target nodes
            edge_index_with_loops, _ = add_self_loops(
                edge_index, 
                num_nodes=num_tgt_nodes
            )
            
            # Create edge attributes for self-loops
            original_edge_attr = edge_attr_dict[edge_type]
            edge_dim = original_edge_attr.size(1)
            
            # Create zero edge attributes for self-loops
            num_self_loops = num_tgt_nodes
            self_loop_attrs = torch.zeros(num_self_loops, edge_dim, device=original_edge_attr.device)
            
            # Concatenate original edge attributes with self-loop attributes
            all_edge_attrs = torch.cat([original_edge_attr, self_loop_attrs], dim=0)
            
            # Sort edge indices by destination nodes for LSTM aggregation
            edge_index_sorted, edge_attr_sorted = sort_edge_index(
                edge_index_with_loops, 
                all_edge_attrs, 
                sort_by_row=False  # Sort by destination nodes (col)
            )
            
            edge_index_dict_with_self_loops[edge_type] = edge_index_sorted
            edge_attr_dict_with_self_loops[edge_type] = edge_attr_sorted
        
        # Ensure all expected edge types have valid tensors (even if empty)
        device = x_encoded['source'].device
        edge_dim = next(iter(edge_attr_dict.values())).size(1) if edge_attr_dict else 1
        
        for edge_type in expected_edge_types:
            if edge_type not in edge_index_dict_with_self_loops:
                # Create empty edge index and attributes for missing edge types
                edge_index_dict_with_self_loops[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr_dict_with_self_loops[edge_type] = torch.empty((0, edge_dim), dtype=torch.float, device=device)
        
        # If no edges exist at all, return empty tensor early
        if not non_empty_edge_types:
            # Create dummy output with correct shape
            dummy_device = x_encoded['source'].device
            return torch.empty((0, 2), device=dummy_device)
        
        # Graph convolution layers with residual connections
        x1 = self.conv1(x_encoded, edge_index_dict_with_self_loops)
        x1 = {k: self.relu(v) for k, v in x1.items()}
        x1 = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x1.items()}
        
        x2 = self.conv2(x1, edge_index_dict_with_self_loops)
        x2 = {k: self.relu(v + x1[k]) for k, v in x2.items()}  # Residual connection
        x2 = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x2.items()}
        
        x3 = self.conv3(x2, edge_index_dict_with_self_loops)
        x3 = {k: self.relu(v + x2[k]) for k, v in x3.items()}  # Residual connection
        x3 = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x3.items()} 