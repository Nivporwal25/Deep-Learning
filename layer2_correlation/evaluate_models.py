import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np

from alert_encoder import IMMUNEX_AlertTransformer
from temporal_models import TemporalBiLSTM
from train_bilstm import build_bilstm_dataset

def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===============================================================")
    print(f"Evaluating God-Mode Models on {str(device).upper()}")
    print(f"===============================================================")
    
    # 1. Evaluate Transformer Spatial Node Accuracy
    data_path = os.path.join("data", "l2_seq_dataset.pt")
    if not os.path.exists(data_path):
        print(f"[-] Dataset missing: {data_path}")
        return
        
    dataset_tensors = torch.load(data_path, map_location=device, weights_only=True)
    features = dataset_tensors['features']
    nodes = dataset_tensors['nodes']
    
    transformer_path = os.path.join("models", "transformer", "immunex_transformer_godmode.pt")
    transformer = IMMUNEX_AlertTransformer(input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4).to(device)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    transformer.eval()
    
    print("[+] Evaluating Transformer Spatial Node Classification...")
    correct_nodes = 0
    total_nodes = 0
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            b_x = features[i:i+batch_size].to(device)
            b_y = nodes[i:i+batch_size].to(device)
            
            nodes_pred, _, _, _ = transformer(b_x) # (B, 50, 4)
            preds = torch.argmax(nodes_pred, dim=2) # (B, 50)
            
            correct_nodes += (preds == b_y).sum().item()
            total_nodes += b_y.numel()
            
    transformer_acc = (correct_nodes / total_nodes) * 100
    print(f"    -> Final Transformer Node Accuracy: {transformer_acc:.2f}%\n")
    
    # 2. Evaluate BiLSTM Narrative Accuracy
    print("[+] Evaluating Temporal BiLSTM Predictive Tracker...")
    seq_features, seq_labels = build_bilstm_dataset(device, seq_len=10)
    
    if seq_features is None:
        return
        
    bilstm_path = os.path.join("models", "bilstm", "immunex_bilstm_godmode.pt")
    bilstm = TemporalBiLSTM(input_size=118, hidden_size=128, num_layers=2, num_classes=5).to(device)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location=device, weights_only=True))
    bilstm.eval()
    
    correct_seqs = 0
    total_seqs = 0
    
    with torch.no_grad():
        for i in range(0, seq_features.shape[0], batch_size):
            b_x = seq_features[i:i+batch_size].to(device)
            b_y = seq_labels[i:i+batch_size].to(device)
            
            probs = bilstm(b_x)
            preds = torch.argmax(probs, dim=1)
            
            correct_seqs += (preds == b_y).sum().item()
            total_seqs += b_y.size(0)
            
    bilstm_acc = (correct_seqs / total_seqs) * 100
    print(f"    -> Final BiLSTM Narrative Accuracy: {bilstm_acc:.2f}%")

if __name__ == "__main__":
    evaluate_models()
