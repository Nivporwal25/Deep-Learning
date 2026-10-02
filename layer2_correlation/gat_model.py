import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, to_hetero, global_mean_pool
from torch_geometric.nn import Linear as PyGLinear

class BaseGATv2(nn.Module):
    def __init__(self, hidden_channels=64, heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # 3-Layer GATv2 with skip connections to prevent over-smoothing
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
        self.skip1 = PyGLinear(-1, hidden_channels * heads)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
        self.skip2 = PyGLinear(-1, hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        
        # Final layer compresses heads (concat=False)
        self.conv3 = GATv2Conv((-1, -1), hidden_channels, heads=heads, concat=False, add_self_loops=False)
        self.skip3 = PyGLinear(-1, hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.skip1(x)
        x = self.norm1(x).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index) + self.skip2(x)
        x = self.norm2(x).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index) + self.skip3(x)
        x = self.norm3(x).relu()
        return x

class IMMUNEX_GATv2_Hetero(nn.Module):
    def __init__(self, metadata, hidden_channels=128, num_classes=4, out_dim=118, heads=8, dropout=0.3):
        super().__init__()
        
        self.gnn = to_hetero(BaseGATv2(hidden_channels, heads, dropout), metadata, aggr='sum')
        
        # Node Classifier: Maps alert embeddings to 4 MITRE stages
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # DQN Blast Radius Compressor: Projects pooled alerts to dense 109D vector
        self.compression_head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
        
        self.severity_predictor = nn.Linear(out_dim, 1)

    def forward(self, x_dict, edge_index_dict, alert_batch=None):
        out_dict = self.gnn(x_dict, edge_index_dict)
        alert_x = out_dict['alert']
        
        node_logits = self.node_classifier(alert_x)
        
        alert_batch = alert_batch if alert_batch is not None else torch.zeros(alert_x.size(0), dtype=torch.long, device=alert_x.device)
        
        graph_repr = global_mean_pool(alert_x, alert_batch)
        state_vector = self.compression_head(graph_repr)
        severity_score = self.severity_predictor(state_vector)
        
        return node_logits, state_vector, severity_score