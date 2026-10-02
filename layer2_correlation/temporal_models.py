import torch
import torch.nn as nn
import numpy as np

class TemporalBiLSTM(nn.Module):
    """
    Phase 2: Temporal Narrative Tracking
    Processes a sequence of 118D Spatial Vectors from the Transformer
    to understand the evolution of the attacker's TTPs over time.
    """
    def __init__(self, input_size=118, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Bidirectional outputs hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len=10, 118D)
        Returns: (Batch, 5D) Softmax probabilities for the Current MITRE Stage
        """
        # out shape: (Batch, Seq_Len, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Extract the final hidden state of the sequence
        last_hidden = out[:, -1, :] 
        
        logits = self.classifier(last_hidden)
        current_stage_probs = torch.softmax(logits, dim=1)
        
        return current_stage_probs

class PredictiveHMM:
    """
    Phase 3: The Predictive Engine
    Using the calculated probabilities of the current stage, this infers
    the attacker's next move using a transition matrix modeled on APT behaviors.
    """
    def __init__(self, transition_matrix_path=None):
        # Default APT Transition Matrix (5x5)
        # Rows = Current State, Cols = Next State
        # Stages: 0=Recon, 1=Access, 2=PrivEsc, 3=Lateral, 4=Exfil
        self.transition_matrix = np.array([
            [0.10, 0.80, 0.05, 0.05, 0.00], # Recon -> Access
            [0.05, 0.10, 0.50, 0.30, 0.05], # Access -> PrivEsc/Lateral
            [0.00, 0.05, 0.20, 0.60, 0.15], # PrivEsc -> Lateral
            [0.00, 0.00, 0.05, 0.15, 0.80], # Lateral -> Exfil
            [0.00, 0.00, 0.00, 0.00, 1.00]  # Exfil -> Sink state
        ], dtype=np.float32)
        
        if transition_matrix_path:
            try:
                self.transition_matrix = np.load(transition_matrix_path)
            except Exception as e:
                print(f"Warning: Could not load HMM matrix, using default. {e}")
                
        self.transition_tensor = torch.tensor(self.transition_matrix)

    def predict_future_stage(self, current_probs: torch.Tensor):
        """
        Calculates the probability distribution of the attacker's NEXT stage.
        current_probs: (Batch, 5)
        Returns: future_probs (Batch, 5)
        """
        self.transition_tensor = self.transition_tensor.to(current_probs.device)
        # Simple Matrix Multiplication: Future Probs = Current Probs * T
        future_probs = torch.matmul(current_probs, self.transition_tensor)
        return future_probs
