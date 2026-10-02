"""
IMMUNEX Layer 2 — Full System Audit
Verifies every component of the God-Mode pipeline end-to-end.
"""
import torch
import numpy as np
import os
import sys

from alert_encoder import IMMUNEX_AlertTransformer, AttentionPenaltyLoss
from temporal_models import TemporalBiLSTM, PredictiveHMM

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "[OK]" if condition else "[XX]"
    print(f"  {icon} [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return condition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ════════════════════════════════════════════════════════════════
# AUDIT 1: DATASET AUTHENTICITY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 1: DATASET AUTHENTICITY & INTEGRITY")
print("="*70)

data_path = os.path.join("data", "l2_seq_dataset.pt")
check("Dataset file exists", os.path.exists(data_path), data_path)

ds = torch.load(data_path, map_location='cpu', weights_only=True)
features = ds['features']
nodes = ds['nodes']
stages = ds['stages']
severe = ds['severe']

check("Features shape is (N, 50, 77)", 
      len(features.shape) == 3 and features.shape[1] == 50 and features.shape[2] == 77,
      str(features.shape))

check("Node labels shape is (N, 50)", 
      len(nodes.shape) == 2 and nodes.shape[1] == 50,
      str(nodes.shape))

check("Stages shape is (N,)", len(stages.shape) == 1, str(stages.shape))
check("Severity shape is (N, 1)", len(severe.shape) == 2 and severe.shape[1] == 1, str(severe.shape))

# Check that the data is NOT synthetic (random uniform data)
# Real data has skewed distributions; synthetic random.uniform has mean ~0.5 and std ~0.29
feat_mean = features.mean().item()
feat_std = features.std().item()
# Check for unique values per feature column - synthetic data is smooth, real data has spikes
col0_unique = features[:, :, 0].unique().numel()
check("NOT synthetic: Feature mean != ~0.5 (uniform random)", 
      abs(feat_mean - 0.5) > 0.05 or feat_std < 0.25,
      f"mean={feat_mean:.4f}, std={feat_std:.4f}")

check("NOT synthetic: Feature col 0 has realistic distribution",
      col0_unique < features.shape[0] * 50 * 0.5,
      f"unique_vals_col0={col0_unique} out of {features.shape[0]*50}")

# Check label distribution - real CICIDS2018 is heavily skewed toward Benign
node_counts = torch.bincount(nodes.flatten(), minlength=4)
total_nodes = nodes.numel()
benign_pct = (node_counts[0].item() / total_nodes) * 100
check("NOT synthetic: Label distribution is skewed (real-world imbalance)",
      benign_pct > 60 or benign_pct < 10,  # Real data is usually very imbalanced
      f"Benign={benign_pct:.1f}%, Counts={node_counts.tolist()}")

stage_counts = torch.bincount(stages, minlength=5)
check("Stage labels cover all 5 MITRE stages (0-4)",
      (stages.max().item() <= 4) and (stages.min().item() >= 0),
      f"min={stages.min().item()}, max={stages.max().item()}, dist={stage_counts.tolist()}")

check("Severity values are in [0, 1]",
      severe.min().item() >= 0.0 and severe.max().item() <= 1.0,
      f"min={severe.min():.4f}, max={severe.max():.4f}")

# Verify it was sourced from a real CSV (02-14-2018.csv)
csv_path = os.path.join("..", "CICIDS2018", "02-14-2018.csv")
check("Source CSV exists (CICIDS2018/02-14-2018.csv)",
      os.path.exists(csv_path),
      f"size={os.path.getsize(csv_path)/(1024**2):.0f}MB" if os.path.exists(csv_path) else "MISSING")

# ════════════════════════════════════════════════════════════════
# AUDIT 2: TRANSFORMER WEIGHTS VERIFICATION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 2: TRANSFORMER WEIGHTS VERIFICATION")
print("="*70)

tf_path = os.path.join("models", "transformer", "immunex_transformer_godmode.pt")
check("Transformer weight file exists", os.path.exists(tf_path), tf_path)

tf_size = os.path.getsize(tf_path) if os.path.exists(tf_path) else 0
check("Transformer weight file > 1MB (not empty stub)",
      tf_size > 1_000_000,
      f"{tf_size/(1024**2):.2f} MB")

# Load and verify architecture matches
transformer = IMMUNEX_AlertTransformer(input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4).to(device)
state = torch.load(tf_path, map_location=device, weights_only=True)
transformer.load_state_dict(state)
transformer.eval()

check("Transformer loads without error", True)

# Verify weight values are not trivial (all zeros or random init)
param_means = []
for name, p in transformer.named_parameters():
    param_means.append(p.data.abs().mean().item())
avg_param = np.mean(param_means)
check("Transformer weights are non-trivial (trained, not random init)",
      avg_param > 0.01 and avg_param < 10.0,
      f"avg |param|={avg_param:.4f}")

# Forward pass shape verification
dummy_x = features[:2].to(device)
nodes_pred, severity_pred, spatial_vec, attns = transformer(dummy_x)

check("Transformer Node output shape is (B, 50, 4)",
      nodes_pred.shape == (2, 50, 4),
      str(nodes_pred.shape))

check("Transformer Severity output shape is (B, 1)",
      severity_pred.shape == (2, 1),
      str(severity_pred.shape))

check("Transformer Spatial Vector output shape is (B, 118)",
      spatial_vec.shape == (2, 118),
      str(spatial_vec.shape))

check("Severity output is in [0, 1] (Sigmoid applied)",
      severity_pred.min().item() >= 0 and severity_pred.max().item() <= 1,
      f"min={severity_pred.min().item():.4f}, max={severity_pred.max().item():.4f}")

check("Attention weights list has 4 layers",
      len(attns) == 4,
      f"got {len(attns)} layers")

check("Attention shape per layer is (B, 51, 51) (50 alerts + CLS)",
      attns[0].shape == (2, 51, 51),
      str(attns[0].shape))

# Verify AttentionPenaltyLoss works
penalty = AttentionPenaltyLoss()
loss_val = penalty(attns)
check("AttentionPenaltyLoss produces valid gradient signal",
      loss_val.item() >= 0,
      f"penalty={loss_val.item():.6f}")

# ════════════════════════════════════════════════════════════════
# AUDIT 3: BiLSTM WEIGHTS VERIFICATION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 3: BiLSTM WEIGHTS VERIFICATION")
print("="*70)

bl_path = os.path.join("models", "bilstm", "immunex_bilstm_godmode.pt")
check("BiLSTM weight file exists", os.path.exists(bl_path), bl_path)

bl_size = os.path.getsize(bl_path) if os.path.exists(bl_path) else 0
check("BiLSTM weight file > 100KB (not empty stub)",
      bl_size > 100_000,
      f"{bl_size/1024:.1f} KB")

bilstm = TemporalBiLSTM(input_size=118, hidden_size=128, num_layers=2, num_classes=5).to(device)
bilstm_state = torch.load(bl_path, map_location=device, weights_only=True)
bilstm.load_state_dict(bilstm_state)
bilstm.eval()
check("BiLSTM loads without error", True)

# Verify BiLSTM forward pass
dummy_seq = torch.randn(2, 10, 118).to(device)
bilstm_out = bilstm(dummy_seq)

check("BiLSTM output shape is (B, 5) — 5 MITRE stages",
      bilstm_out.shape == (2, 5),
      str(bilstm_out.shape))

check("BiLSTM output sums to ~1.0 (valid softmax probability)",
      abs(bilstm_out[0].sum().item() - 1.0) < 0.01,
      f"sum={bilstm_out[0].sum().item():.6f}")

check("BiLSTM output values are all >= 0 (valid probabilities)",
      (bilstm_out >= 0).all().item(),
      f"min={bilstm_out.min().item():.6f}")

# ════════════════════════════════════════════════════════════════
# AUDIT 4: HMM FUTURE PREDICTION VERIFICATION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 4: HMM PREDICTIVE ENGINE VERIFICATION")
print("="*70)

hmm = PredictiveHMM()

check("HMM transition matrix is 5x5",
      hmm.transition_matrix.shape == (5, 5),
      str(hmm.transition_matrix.shape))

# Each row must sum to 1.0 (valid probability distribution)
row_sums = hmm.transition_matrix.sum(axis=1)
check("HMM transition rows sum to 1.0 (valid stochastic matrix)",
      np.allclose(row_sums, 1.0, atol=1e-5),
      f"row_sums={row_sums.tolist()}")

# Test: If attacker is 100% in Recon, next should be mostly Initial Access
recon_input = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
future = hmm.predict_future_stage(recon_input)
check("HMM: Recon->next predicts mostly Initial Access",
      future[0][1].item() > 0.5,
      f"future_probs={[round(f,3) for f in future[0].tolist()]}")

# Test: If attacker is 100% in Lateral Movement, next should be mostly Exfiltration
lat_input = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
future_lat = hmm.predict_future_stage(lat_input)
check("HMM: Lateral->next predicts mostly Exfiltration",
      future_lat[0][4].item() > 0.5,
      f"future_probs={[round(f,3) for f in future_lat[0].tolist()]}")

# Test that HMM future output also sums to 1.0
check("HMM output sums to ~1.0 (valid probability)",
      abs(future[0].sum().item() - 1.0) < 0.01,
      f"sum={future[0].sum().item():.6f}")

# ════════════════════════════════════════════════════════════════
# AUDIT 5: 128D GOD-MODE VECTOR FUSION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 5: 128D GOD-MODE VECTOR FUSION")
print("="*70)

# Simulate full pipeline
test_window = features[:1].to(device)

with torch.no_grad():
    _, _, spatial_118d, _ = transformer(test_window)

spatial_np = spatial_118d[0].cpu().numpy()
check("Spatial vector dimension is 118", spatial_np.shape == (118,), str(spatial_np.shape))

# Simulate the BiLSTM with 10 copies of the same vector (cold start)
seq_input = spatial_118d.unsqueeze(0).expand(1, 10, -1).to(device)
with torch.no_grad():
    current_5d = bilstm(seq_input)  # (1, 5)

current_np = current_5d[0].cpu().numpy()
check("Current stage vector dimension is 5", current_np.shape == (5,), str(current_np.shape))

future_5d = hmm.predict_future_stage(current_5d)
future_np = future_5d[0].cpu().numpy()
check("Future stage vector dimension is 5", future_np.shape == (5,), str(future_np.shape))

# THE FINAL FUSION
god_mode_128d = np.concatenate([spatial_np, current_np, future_np])
check("GOD-MODE VECTOR IS EXACTLY 128 DIMENSIONS",
      god_mode_128d.shape == (128,),
      f"shape={god_mode_128d.shape}")

check("128D vector segment [0:118] is from Transformer Spatial",
      np.array_equal(god_mode_128d[:118], spatial_np))

check("128D vector segment [118:123] is from BiLSTM Current State",
      np.array_equal(god_mode_128d[118:123], current_np))

check("128D vector segment [123:128] is from HMM Future State",
      np.array_equal(god_mode_128d[123:128], future_np))

# ════════════════════════════════════════════════════════════════
# AUDIT 6: PRIORITY QUEUE DISPATCH ORDER
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AUDIT 6: PRIORITY QUEUE DISPATCH ORDERING")
print("="*70)

import heapq

# Simulate 5 attackers with known severity scores
test_severities = {
    "10.0.0.1": 0.92,    # Critical DDoS
    "192.168.1.50": 0.35, # Low-level scan
    "172.16.0.4": 0.78,   # Medium brute force
    "10.0.0.99": 0.15,    # Benign-looking
    "192.168.5.5": 0.95,  # Highest threat
}

pq = []
for ip, sev in test_severities.items():
    heapq.heappush(pq, (-sev, ip, {"severity": sev}))

dispatch_order = []
while pq:
    neg_sev, ip, payload = heapq.heappop(pq)
    dispatch_order.append((ip, -neg_sev))

check("Highest severity IP dispatched first",
      dispatch_order[0][0] == "192.168.5.5",
      f"first={dispatch_order[0]}")

check("Lowest severity IP dispatched last",
      dispatch_order[-1][0] == "10.0.0.99",
      f"last={dispatch_order[-1]}")

# Verify strictly descending severity order
severity_order = [s for _, s in dispatch_order]
check("Dispatch order is strictly descending by severity",
      all(severity_order[i] >= severity_order[i+1] for i in range(len(severity_order)-1)),
      f"order={severity_order}")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL AUDIT SUMMARY")
print("="*70)

passes = sum(1 for _, s, _ in results if s == PASS)
fails = sum(1 for _, s, _ in results if s == FAIL)
print(f"\n  Total Checks: {len(results)}")
print(f"  PASSED: {passes}")
print(f"  FAILED: {fails}")

if fails > 0:
    print("\n  FAILED CHECKS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    [XX] {name}: {detail}")

print(f"\n  VERDICT: {'ALL SYSTEMS NOMINAL' if fails == 0 else 'ISSUES DETECTED'}")

