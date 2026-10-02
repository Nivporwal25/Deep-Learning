"""
IMMUNEX Graph Builder — Phase 1: Graph Construction
Converts raw alert windows into PyG HeteroData graphs.
All edge construction is fully VECTORIZED using numpy/pandas — eliminates
the nested Python O(n²) loop that was causing 0% GPU utilization.
"""
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from typing import List, Dict

# The actual numeric columns from the UNSW-NB15 dataset (already standardized by data_processor.py)
ALERT_FEATURE_COLS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    'proto_encoded', 'state_encoded'
]

# 4-class MITRE mapping (Worms merged into Install due to only 44 samples)
MITRE_STAGE_MAP = {
    'Reconnaissance': 0, 'Analysis': 0,       # Stage 0: Recon
    'Fuzzers': 1, 'Exploits': 1,              # Stage 1: Exploit
    'Backdoors': 2, 'Backdoor': 2,            # Stage 2: Install (handles both spellings)
    'Shellcode': 2, 'Worms': 2,
    'DoS': 3, 'Generic': 3                    # Stage 3: Action
}


def _build_blast_radius_edges_vectorized(src_ips: np.ndarray, dst_ips: np.ndarray, users: np.ndarray):
    """
    Fully vectorized blast-radius edge builder — replaces the O(n²) Python loop.
    Finds all pairs (i, j) where i < j that share a source IP, dest IP, or username.
    Returns bidirectional [src_list, dst_list] tensors.
    """
    n = len(src_ips)
    src_list, dst_list = [], []

    # Group alert indices by shared source IP
    for group_col in [src_ips, dst_ips, users]:
        groups = {}
        for idx, val in enumerate(group_col):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val_str = str(val).strip()
            if val_str in ('', 'nan', 'None', 'unknown', 'unknown.exe'):
                continue
            groups.setdefault(val_str, []).append(idx)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            arr = np.array(indices)
            # Create all unique pairs using broadcasting
            ii, jj = np.triu_indices(len(arr), k=1)
            a, b = arr[ii], arr[jj]
            # Bidirectional
            src_list.extend(a.tolist())
            dst_list.extend(b.tolist())
            src_list.extend(b.tolist())
            dst_list.extend(a.tolist())

    # Deduplicate edges efficiently
    if len(src_list) == 0:
        return [], []

    edges = np.unique(np.column_stack([src_list, dst_list]), axis=0)
    return edges[:, 0].tolist(), edges[:, 1].tolist()


def construct_heterogeneous_graph(events: List[Dict]) -> HeteroData:
    df = pd.DataFrame(events)

    if 'source_ip' not in df.columns:
        df['source_ip'] = [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(len(df))]
    if 'dest_ip' not in df.columns:
        df['dest_ip'] = "10.0.50.100"

    data = HeteroData()

    # --- Node Extraction ---
    src_ips = df['source_ip'].replace(["", "unknown", "None"], np.nan)
    dst_ips = df['dest_ip'].replace(["", "unknown", "None"], np.nan)
    file_col = df['file'].replace(["", "none", "None"], np.nan)
    user_col = df['username'].replace(["", "unknown", "None"], np.nan)
    proc_col = df['process'].replace(["", "unknown.exe", "None"], np.nan)

    all_ips = pd.concat([src_ips, dst_ips]).dropna().unique()
    unique_files = file_col.dropna().unique()
    unique_users = user_col.dropna().unique()
    unique_procs = proc_col.dropna().unique()

    ip_to_id = {ip: i for i, ip in enumerate(all_ips)}
    file_to_id = {f: i for i, f in enumerate(unique_files)}
    user_to_id = {u: i for i, u in enumerate(unique_users)}
    proc_to_id = {p: i for i, p in enumerate(unique_procs)}

    num_alerts = len(df)

    # --- Alert Features (vectorized) ---
    available_cols = [c for c in ALERT_FEATURE_COLS if c in df.columns]
    alert_features = df[available_cols].fillna(0).values.astype(np.float32) if available_cols else \
        np.zeros((num_alerts, len(ALERT_FEATURE_COLS)), dtype=np.float32)

    # --- Labels (vectorized) ---
    alert_labels = df['attack_cat'].fillna('Normal').astype(str).str.strip().map(
        lambda x: MITRE_STAGE_MAP.get(x, -1)
    ).values.astype(np.int64)

    data['alert'].x = torch.tensor(alert_features, dtype=torch.float)
    data['alert'].y = torch.tensor(alert_labels, dtype=torch.long)

    # Entity node features
    if len(all_ips) > 0:      data['machine'].x = torch.ones((len(all_ips), 4), dtype=torch.float)
    if len(unique_files) > 0: data['file'].x    = torch.ones((len(unique_files), 2), dtype=torch.float)
    if len(unique_users) > 0: data['user'].x    = torch.ones((len(unique_users), 2), dtype=torch.float)
    if len(unique_procs) > 0: data['process'].x = torch.ones((len(unique_procs), 2), dtype=torch.float)

    is_malicious = float((alert_labels >= 0).any())
    data.y_graph = torch.tensor([[is_malicious]], dtype=torch.float)

    # --- Vectorized Entity Edge Building ---
    edges = {
        ('alert', 'originates_from', 'machine'): [[], []],
        ('alert', 'targets', 'machine'): [[], []],
        ('alert', 'involves', 'file'): [[], []],
        ('alert', 'executed', 'process'): [[], []],
        ('alert', 'performed_by', 'user'): [[], []],
        ('alert', 'blast_radius', 'alert'): [[], []]
    }

    alert_indices = np.arange(num_alerts)

    # Source IP edges
    mapped = src_ips.map(ip_to_id).dropna()
    if len(mapped) > 0:
        valid_idx = mapped.index.tolist()
        edges[('alert', 'originates_from', 'machine')][0] = valid_idx
        edges[('alert', 'originates_from', 'machine')][1] = mapped.values.astype(int).tolist()

    # Dest IP edges
    mapped = dst_ips.map(ip_to_id).dropna()
    if len(mapped) > 0:
        valid_idx = mapped.index.tolist()
        edges[('alert', 'targets', 'machine')][0] = valid_idx
        edges[('alert', 'targets', 'machine')][1] = mapped.values.astype(int).tolist()

    # File edges
    mapped = file_col.map(file_to_id).dropna()
    if len(mapped) > 0:
        edges[('alert', 'involves', 'file')][0] = mapped.index.tolist()
        edges[('alert', 'involves', 'file')][1] = mapped.values.astype(int).tolist()

    # Process edges
    mapped = proc_col.map(proc_to_id).dropna()
    if len(mapped) > 0:
        edges[('alert', 'executed', 'process')][0] = mapped.index.tolist()
        edges[('alert', 'executed', 'process')][1] = mapped.values.astype(int).tolist()

    # User edges
    mapped = user_col.map(user_to_id).dropna()
    if len(mapped) > 0:
        edges[('alert', 'performed_by', 'user')][0] = mapped.index.tolist()
        edges[('alert', 'performed_by', 'user')][1] = mapped.values.astype(int).tolist()

    # Blast-Radius Edges — VECTORIZED (replaces the O(n²) Python loop)
    br_src, br_dst = _build_blast_radius_edges_vectorized(
        src_ips.values, dst_ips.values, user_col.values
    )
    edges[('alert', 'blast_radius', 'alert')] = [br_src, br_dst]

    for edge_type, (src_list, dst_list) in edges.items():
        if len(src_list) > 0:
            data[edge_type].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    data = T.ToUndirected()(data)
    return data


def generate_neo4j_pre_approval_commit(events: List[Dict], attack_uuid: str) -> List[tuple]:
    """
    Phase 3 (Stage 1): Pre-Approval Commit — writes graph to Neo4j for the dashboard.

    SEC 1 fix: all Cypher queries are now PARAMETERIZED (query, params dict) tuples.
    Raw f-string queries allowed Cypher injection via attacker-controlled source_ip/file/username.
    Caller must use session.run(query, **params) — see commit_to_neo4j in integration_pipeline.py.
    """
    df = pd.DataFrame(events)
    # Each entry: (cypher_string, params_dict)
    queries: List[tuple] = [
        (
            "MERGE (i:Incident {incident_id: $incident_id}) SET i.status = 'AWAITING_APPROVAL'",
            {"incident_id": attack_uuid}
        )
    ]

    for _, row in df.iterrows():
        alert_id = row.get("alert_id", f"A_{int(np.random.rand()*100000)}")
        src_ip   = row.get('source_ip')
        file_acc = row.get('file')
        user     = row.get('username')
        evt_type = row.get('event_type')

        queries.append((
            "MERGE (a:Alert {alert_id: $alert_id}) SET a.event_type = $evt_type",
            {"alert_id": str(alert_id), "evt_type": str(evt_type) if evt_type else ""}
        ))
        queries.append((
            "MATCH (i:Incident {incident_id: $incident_id}), (a:Alert {alert_id: $alert_id}) "
            "MERGE (i)-[:CONTAINS]->(a)",
            {"incident_id": attack_uuid, "alert_id": str(alert_id)}
        ))

        if src_ip and str(src_ip).lower() not in ['', 'unknown', 'none']:
            queries.append((
                "MERGE (m:Machine {ip: $ip})",
                {"ip": str(src_ip)}
            ))
            queries.append((
                "MATCH (a:Alert {alert_id: $alert_id}), (m:Machine {ip: $ip}) "
                "MERGE (a)-[:ORIGINATES_FROM]->(m)",
                {"alert_id": str(alert_id), "ip": str(src_ip)}
            ))

        if file_acc and str(file_acc).lower() not in ['', 'none']:
            queries.append((
                "MERGE (f:File {path: $path})",
                {"path": str(file_acc)}
            ))
            queries.append((
                "MATCH (a:Alert {alert_id: $alert_id}), (f:File {path: $path}) "
                "MERGE (a)-[:INVOLVES]->(f)",
                {"alert_id": str(alert_id), "path": str(file_acc)}
            ))

        if user and str(user).lower() not in ['', 'unknown', 'none']:
            queries.append((
                "MERGE (u:User {username: $username})",
                {"username": str(user)}
            ))
            queries.append((
                "MATCH (a:Alert {alert_id: $alert_id}), (u:User {username: $username}) "
                "MERGE (a)-[:PERFORMED_BY]->(u)",
                {"alert_id": str(alert_id), "username": str(user)}
            ))

    return queries


def generate_neo4j_post_approval_commit(attack_uuid: str, action: str, admin: str, status: str = "COMPLETED") -> str:
    """Phase 3 (Stage 2): Post-Approval Commit — creates audit trail Response node."""
    return (f"MATCH (i:Incident {{incident_id: '{attack_uuid}'}}) "
            f"CREATE (r:Response {{action: '{action}', approved_by: '{admin}', status: '{status}'}}) "
            f"CREATE (i)-[:RESPONDED_WITH]->(r) "
            f"SET i.status = '{status}'")


def create_hetero_graph_dataset(events: List[Dict], window_size: int = 50) -> List[HeteroData]:
    graphs = []
    for i in range(0, len(events), window_size):
        chunk = events[i: i + window_size]
        if len(chunk) > 0:
            graphs.append(construct_heterogeneous_graph(chunk))
    return graphs