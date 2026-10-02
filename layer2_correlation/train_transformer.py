import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os

from alert_encoder import IMMUNEX_AlertTransformer, AttentionPenaltyLoss

def train_transformer(epochs=10, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===============================================================")
    print(f"STAGE 1: Training God-Mode Spatial Alert Transformer")
    print(f"Execution Context Binding: STRICTLY {str(device).upper()}")
    print(f"===============================================================")
    
    if str(device) == 'cpu':
        print("WARNING: CUDA is not available. Training will be extremely slow.")

    data_path = os.path.join("data", "l2_seq_dataset.pt")
    if not os.path.exists(data_path):
        print(f"[-] CRITICAL FAILURE: Dataset not found at {data_path}")
        return

    # Load dataset tensors entirely onto the target device if they fit, 
    # but the 94MB tensor will easily fit on GPU memory.
    print("[+] Loading Phase 0 sequence dataset into VRAM...")
    dataset_tensors = torch.load(data_path, map_location=device, weights_only=True)
    
    features = dataset_tensors['features']  # (Batch, 50, 77)
    nodes = dataset_tensors['nodes']        # (Batch, 50) Node classification targets
    # stages = dataset_tensors['stages']    # (Batch,) Used in BiLSTM, not here
    severe = dataset_tensors['severe']      # (Batch, 1) Window severity target

    print(f"[+] Spatial Tensor Matrix Loaded. Batch Count: {features.shape[0]}, Sequence Limit: 50, Feature Depth: 77")
    
    # Construct PyTorch Dataset and split Train/Val
    dataset = TensorDataset(features, nodes, severe)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("[+] Compiling IMMUNEX_AlertTransformer (4-Layer, 8-Head)")
    model = IMMUNEX_AlertTransformer(input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4).to(device)

    # Multi-task Loss configuration
    node_criterion = nn.CrossEntropyLoss()
    severe_criterion = nn.MSELoss()
    attn_penalty_criterion = AttentionPenaltyLoss(min_threshold=1e-3, penalty_weight=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    model_save_dir = os.path.join("models", "transformer")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "immunex_transformer_godmode.pt")

    print("\n[+] Accelerating Pipeline... Commencing Epochs:")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_node_loss = 0.0
        train_sev_loss = 0.0
        train_attn_pen = 0.0
        
        for batch_idx, (b_x, b_nodes, b_sev) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward God-Mode Head Branching
            nodes_pred, sev_pred, spatial_vec, attns = model(b_x)
            
            # Loss 1: Nodes (flatten batch and sequence lengths)
            # nodes_pred shape is (B, 50, 4) -> (B*50, 4)
            # b_nodes shape is (B, 50) -> (B*50,)
            loss_nodes = node_criterion(nodes_pred.reshape(-1, 4), b_nodes.reshape(-1))
            
            # Loss 2: Severity (B, 1)
            loss_sev = severe_criterion(sev_pred, b_sev)
            
            # Loss 3: Custom Mandatory Attention Constraint
            loss_attn = attn_penalty_criterion(attns)
            
            # Total objective function
            loss = loss_nodes + loss_sev + loss_attn
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_node_loss += loss_nodes.item()
            train_sev_loss += loss_sev.item()
            train_attn_pen += loss_attn.item()
            
        # Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_node_loss = train_node_loss / len(train_loader)
        avg_sev_loss = train_sev_loss / len(train_loader)
        avg_attn_pen = train_attn_pen / len(train_loader)
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x, b_nodes, b_sev in val_loader:
                nodes_pred, sev_pred, _, attns = model(b_x)
                loss_nodes = node_criterion(nodes_pred.reshape(-1, 4), b_nodes.reshape(-1))
                loss_sev = severe_criterion(sev_pred, b_sev)
                loss_attn = attn_penalty_criterion(attns)
                loss = loss_nodes + loss_sev + loss_attn
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Total: {avg_train_loss:.4f} "
              f"| NodeLoss: {avg_node_loss:.4f} "
              f"| SevLoss: {avg_sev_loss:.4f} "
              f"| AttnPen: {avg_attn_pen:.4f} "
              f"--- Val Loss: {avg_val_loss:.4f}")
              
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Create subfolder and save model dict
            torch.save(model.state_dict(), model_save_path)
    
    print(f"\n[+] Transformer Model Weights persisted to {model_save_path}")
    print("[+] PHASE 1 SPATIAL TRAINING COMPLETED.")

if __name__ == "__main__":
    train_transformer(epochs=10)
