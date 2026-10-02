import torch
import torch.nn as nn

class IMMUNEX_BiLSTM_Tracker(nn.Module):
    def __init__(self, input_dim=118, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3):
        """
        The BiLSTM Narrative Tracker.
        Takes a chronological sequence of 118D spatial vectors from the GATv2
        and predicts the current phase of the attack in the 5-class MITRE kill-chain.
        
        Args:
            input_dim: 118 (Output from GATv2 Compression Head — the Spatial Vector)
            hidden_dim: The memory capacity of the LSTM cells
            num_layers: Number of stacked BiLSTM layers
            num_classes: 5 (Recon, Initial Access, Priv Esc, Lat Movement, Exfil)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM to evaluate the past -> present AND the future -> past
        # batch_first=True expects input shape: (batch_size, sequence_length, input_dim)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier Head
        # Since it's bidirectional, the output from the LSTM is 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, 118)
        Returns: logits of shape (batch_size, 5) representing the prediction 
                 for the MOST RECENT (current) stage in the chronological sequence.
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # We only care about the classification of the CURRENT stage, 
        # which is the output corresponding to the final time step in the sequence.
        current_state_features = lstm_out[:, -1, :]
        
        # Logits for the 5 MITRE stages
        logits = self.classifier(current_state_features)
        
        return logits
