"""
IMMUNEX GATv2 — Phase 2 Training Script (GPU Enforced)
Trains the heterogeneous GATv2 model on real UNSW-NB15 features for 4-class MITRE classification.

GPU Optimization checklist:
  ✅ Strict CUDA enforcement — crashes immediately if wrong Python version used
  ✅ pin_memory=True + num_workers=0 (safe for Windows)
  ✅ Vectorized graph construction (graph_builder.py) — CPU no longer bottlenecks GPU
  ✅ torch.cuda.amp (Automatic Mixed Precision) — 1.5–2x speedup on RTX 4050
  ✅ OneCycleLR scheduler for 50-epoch convergence
  ✅ Gradient clipping and inverse-frequency class weights
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
import os
from torch_geometric.loader import DataLoader

from graph_builder import create_hetero_graph_dataset
from gat_model import IMMUNEX_GATv2_Hetero

def train_immunex_gatPipeline(
    csv_path: str = "immunex_final_dataset.csv",
    epochs: int = 50,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    save_path: str = "immunex_gatv2_phase2.pt"
):
    # ------------------------------------------------------------------ GPU CHECK
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CRITICAL: CUDA GPU not detected! You are likely running this script using the wrong Python version.\n"
            "Do NOT run `python train_gat.py`.\n"
            "Instead, run exactly: `py -3.14 train_gat.py` to use the CUDA-enabled environment."
        )
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True   # Enables cuDNN autotuner for fastest kernel
    print(f"--- Launching IMMUNEX GATv2 Training STRICTLY on {torch.cuda.get_device_name(0)} ---")
    print(f"    VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if not os.path.exists(csv_path):
        print(f"Dataset load failed. '{csv_path}' not found. Make sure to run data_processor.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} telemetry events from {csv_path}")

    # 1. Sliding Window Construction (Vectorized — no Python loops blocking GPU)
    print("Slicing 50-alert Sliding Window heterogeneous graphs...")
    events = df.to_dict('records')
    dataset = create_hetero_graph_dataset(events, window_size=50)
    print(f"Constructed {len(dataset)} discrete heterogeneous attack graphs.")

    if len(dataset) == 0:
        return

    # 2. Extract Metadata & Setup DataLoader
    # num_workers=0 is required on Windows to avoid multiprocessing issues with PyG
    metadata = dataset[0].metadata()
    loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)

    # 3. Model & Optimiser
    print("Initialising GATv2 model on GPU...")
    model = IMMUNEX_GATv2_Hetero(
        metadata=metadata, hidden_channels=128, num_classes=4, out_dim=118, heads=8
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP Scaler — uses float16 for forward/backward pass on GPU (much faster on RTX)
    scaler = GradScaler('cuda')

    # 4. Multi-Task Training Criteria
    # Inverse-frequency class weights for 4-class MITRE stages
    class_weights = torch.tensor([3.0, 1.5, 5.0, 1.0], dtype=torch.float, device=device)
    criterion_node = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    criterion_graph = nn.BCEWithLogitsLoss()

    # OneCycleLR for fast convergence in 50 epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader)
    )

    print(f"\nStarting Unified Training Loop ({epochs} epochs, {len(loader)} batches/epoch):")
    best_f1 = 0.0
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_data in loader:
            # .to(device, non_blocking=True) pushes tensors asynchronously while GPU is busy
            batch_data = batch_data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            # AMP autocast: runs forward pass in float16 for ~2x GPU speed
            with autocast('cuda'):
                node_logits, state_vector, severity_score = model(
                    batch_data.x_dict,
                    batch_data.edge_index_dict,
                    batch_data['alert'].batch
                )

                labels = batch_data['alert'].y
                mask = (labels != -1)

                loss_node = criterion_node(node_logits[mask], labels[mask]) if mask.sum() > 0 else \
                    torch.tensor(0.0, device=device, requires_grad=True)

                loss_graph = criterion_graph(severity_score.view(-1), batch_data.y_graph.view(-1).float())
                loss = loss_node + loss_graph

            # AMP backward + optimizer step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

            with torch.no_grad():
                if mask.sum() > 0:
                    preds = node_logits[mask].argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels[mask].cpu().numpy())

        epoch_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        epoch_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        avg_loss  = total_loss / len(loader)

        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            vram_used = torch.cuda.memory_reserved(0) / 1024**3
            print(f"Epoch {epoch:03d}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
                  f"Acc: {epoch_acc*100:.2f}% | F1: {epoch_f1*100:.2f}% | VRAM: {vram_used:.2f}GB")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n--- Training Complete ---")
    print(f"Best Network State Saved: {save_path} (Accuracy: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%)")

if __name__ == "__main__":
    train_immunex_gatPipeline()
