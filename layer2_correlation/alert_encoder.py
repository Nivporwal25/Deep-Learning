import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom transformer layer to expose attention weights for the AttentionPenaltyLoss.
    """
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # We need the alignment weights for the custom penalty
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class IMMUNEX_AlertTransformer(nn.Module):
    def __init__(self, input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4):
        super().__init__()
        self.d_model = d_model
        
        # 1. Linear Projection (77 -> 128)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. The [CLS] Token (Blank Notepad)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=60)
        
        # 4. Transformer Encoder (Custom layers to extract attention)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
            for _ in range(num_layers)
        ])
        
        # 5. Three-way Output Branching
        # Node Logs Head (Classifies each of the 50 alerts)
        self.node_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Severity Score Head (From CLS token)
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Spatial Vector Head (From CLS token -> compresses to 118D)
        self.spatial_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 118)
        )

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Features) e.g. (B, 50, 77)
        Returns:
            nodes_out: (B, 50, 4)
            severity: (B, 1)
            spatial_vec: (B, 118)
            attn_weights: list of (B, Seq_Len+1, Seq_Len+1) tensors for the penalty loss
        """
        B, seq_len, _ = x.shape
        
        # Project raw features
        x = self.input_projection(x) # (B, 50, 128)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, 128)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 51, 128)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer Layers
        layer_attns = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            layer_attns.append(attn_weights)
            
        # The finalized [CLS] token is at index 0
        cls_out = x[:, 0, :] # (B, 128)
        
        # The processed alerts are at indices 1 to end
        nodes_out = x[:, 1:, :] # (B, 50, 128)
        
        # 1. Node Classifier Output
        nodes_pred = self.node_classifier(nodes_out) # (B, 50, 4)
        
        # 2. Severity Score Output
        severity = self.severity_head(cls_out) # (B, 1) float [0-1]
        
        # 3. Spatial Vector
        spatial_vec = self.spatial_head(cls_out) # (B, 118)
        
        return nodes_pred, severity, spatial_vec, layer_attns

class AttentionPenaltyLoss(nn.Module):
    """
    Enforces that across the window, the attention heads do not ignore 'lone-wolf' anomalies.
    Every alert must receive a minimum attention weight threshold from the [CLS] token.
    """
    def __init__(self, min_threshold=1e-3, penalty_weight=0.1):
        super().__init__()
        self.min_threshold = min_threshold
        self.penalty_weight = penalty_weight

    def forward(self, layer_attns):
        # Check attention directed from the [CLS] token to the actual alerts
        total_penalty = 0.0
        for attn in layer_attns:
            # attn shape: (Batch, TargetSeq, SourceSeq) = (B, 51, 51)
            # We want cls_to_alerts attention: (Batch, index=0, source_indices=1:51)
            cls_attn = attn[:, 0, 1:] # Drop self-attention to CLS token itself
            
            # Loss is applied to elements that fall below the minimum threshold
            # F.relu(thresh - value) is positive where value < thresh
            penalty = F.relu(self.min_threshold - cls_attn)
            total_penalty += penalty.mean()
            
        return total_penalty * self.penalty_weight
