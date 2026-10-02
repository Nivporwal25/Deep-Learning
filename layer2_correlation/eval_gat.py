import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch_geometric.loader import DataLoader
from graph_builder import create_hetero_graph_dataset
from gat_model import IMMUNEX_GATv2_Hetero
import warnings
warnings.filterwarnings("ignore")

def evaluate_immunex_gatv2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute: {device}")
    
    print("Loading test telemetry sequences...")
    df = pd.read_csv("immunex_final_dataset.csv")
    events = df.iloc[:15000].to_dict('records')
    dataset = create_hetero_graph_dataset(events, window_size=50)
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    metadata = dataset[0].metadata()
    
    print("Loading baked weights...")
    model = IMMUNEX_GATv2_Hetero(
        metadata=metadata, hidden_channels=128, num_classes=4, out_dim=118, heads=8
    ).to(device)
    model.load_state_dict(torch.load("immunex_gatv2_phase2.pt", map_location=device, weights_only=True))
    model.eval()
    
    all_preds, all_labels = [], []
    print("Evaluating Multi-Agent Graph Nodes...")
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            node_logits, _, _ = model(batch_data.x_dict, batch_data.edge_index_dict, batch_data['alert'].batch)
            
            mask = batch_data['alert'].y != -1
            if mask.sum() > 0:
                preds = torch.argmax(node_logits[mask], dim=1).cpu().numpy()
                labels = batch_data['alert'].y[mask].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\n=======================================================")
    print(f"       IMMUNEX GATv2 MITRE CORE INTELLIGENCE           ")
    print(f"=======================================================")
    print(f"  Overall Node Accuracy       : {acc*100:.2f}%")
    print(f"  Weighted F1-Score (Graph)   : {f1*100:.2f}%")
    print(f"  Heterogeneous Embeddings    : Loaded Successfully")
    print(f"=======================================================")
    
    # Per-class breakdown
    stage_names = ['Recon', 'Exploit', 'Install', 'Action']
    print(f"\nPer-Class Report:")
    print(classification_report(all_labels, all_preds, target_names=stage_names, zero_division=0))

if __name__ == "__main__":
    evaluate_immunex_gatv2()
