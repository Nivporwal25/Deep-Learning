import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np

from alert_encoder import IMMUNEX_AlertTransformer
from temporal_models import TemporalBiLSTM

def build_bilstm_dataset(device, seq_len=10):
    print("[+] Extracting 118D Spatial sequences from Trained Transformer...")
    data_path = os.path.join("data", "l2_seq_dataset.pt")
    
    transformer_path = os.path.join("models", "transformer", "immunex_transformer_godmode.pt")
    if not os.path.exists(transformer_path):
        print(f"[-] CRITICAL: Transformer weights missing at {transformer_path}. Must train Phase 1 first.")
        return None, None
        
    # Dynamically inject the raw data through the frozen Transformer to collect 118D sequences
    transformer = IMMUNEX_AlertTransformer(input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4).to(device)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    transformer.eval()
    
    dataset_tensors = torch.load(data_path, map_location=device, weights_only=True)
    features = dataset_tensors['features'] # (Batch, 50, 77)
    stages = dataset_tensors['stages']     # (Batch) MITRE stages 0-4
    
    print(f"   > Processing {features.shape[0]} raw God-Mode data windows to 118D Spatial Arrays...")
    
    batch_size = 256
    all_118d = []
    
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            b_x = features[i:i+batch_size]
            _, _, spatial_vec, _ = transformer(b_x)
            all_118d.append(spatial_vec)
            
    all_118d = torch.cat(all_118d, dim=0) # (Batch, 118)
    print(f"   > Spatial extraction complete. Transcoding into sequential narratives (len={seq_len})...")
    
    num_samples = all_118d.shape[0]
    seq_features = []
    seq_labels = []
    
    for i in range(0, num_samples - seq_len + 1, seq_len):
        seq = all_118d[i:i+seq_len] # (10, 118)
        # Using the final state of the sequence as the target
        label = stages[i+seq_len-1]
        
        seq_features.append(seq)
        seq_labels.append(label)
        
    if not seq_features:
        print("[-] Error: Dataset too small to build even one sequence array.")
        return None, None
        
    seq_features = torch.stack(seq_features)
    seq_labels = torch.stack(seq_labels)
    
    print(f"   > BiLSTM Database Generated: {seq_features.shape[0]} narrative sequences.")
    return seq_features, seq_labels

def train_bilstm(epochs=15, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===============================================================")
    print(f"STAGE 2: Training Temporal Narrative BiLSTM Tracker")
    print(f"Execution Context Binding: STRICTLY {str(device).upper()}")
    print(f"===============================================================")
    
    features, labels = build_bilstm_dataset(device, seq_len=10)
    if features is None: return
    
    dataset = TensorDataset(features, labels)
    
    train_size = int(0.8 * len(dataset))
    if train_size == 0: train_size = len(dataset) 
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_size > 0 else train_loader
    
    model = TemporalBiLSTM(input_size=118, hidden_size=128, num_layers=2, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    model_save_dir = os.path.join("models", "bilstm")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "immunex_bilstm_godmode.pt")
    
    print("\n[+] Accelerating Pipeline... Commencing BiLSTM Epochs:")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            current_stage_probs = model(b_x) # (Batch, 5)
            loss = criterion(current_stage_probs, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                probs = model(b_x)
                loss = criterion(probs, b_y)
                val_loss += loss.item()
                
                preds = torch.argmax(probs, dim=1)
                correct += (preds == b_y).sum().item()
                total += b_y.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Acc: {accuracy:.1f}%")
              
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            
    # Always save final if no val improvement (small dataset case)
    if train_size == len(dataset):
        torch.save(model.state_dict(), model_save_path)
        
    print(f"\n[+] BiLSTM Model Weights persisted to {model_save_path}")
    print("[+] PHASE 2 NARRATIVE TRAINING COMPLETED.")

if __name__ == "__main__":
    train_bilstm(epochs=15)
