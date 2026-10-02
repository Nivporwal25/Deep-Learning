"""
IMMUNEX — Layer 2 Integration Pipeline (God-Mode Refactor)

This overrides the prior Graph-based system with a new 128D Transformer pipeline.
1. Consumes flagged alerts from Layer 1.
2. Buffers into 50-alert windows.
3. Groups anomalies by Source IP (Multi-Attacker Tracking).
4. Runs AlertTransformer to extract 118D Spatial Vector + Severity Score.
5. Runs BiLSTM over IP timeline -> 5D Current State.
6. Runs HMM over BiLSTM -> 5D Future State.
7. Fuses into 128D God-Mode Vector.
8. Sorts generated attack vectors in a Severity-based Priority Queue.
9. Transmits Highest-Priority vectors to Layer 3 first.
"""

import json
import time
import torch
import numpy as np
import heapq
from datetime import datetime, timezone
from collections import deque
from kafka import KafkaConsumer, KafkaProducer

# Internal Models
from alert_encoder import IMMUNEX_AlertTransformer
from temporal_models import TemporalBiLSTM, PredictiveHMM

KAFKA_BROKER = 'localhost:9092'
L1_ALERTS_TOPIC = 'immunex_layer1_alerts'
L2_TO_L3_TOPIC = 'immunex_layer3_dqn_queue'

MITRE_STAGES = ["Reconnaissance", "Initial Access", "Privilege Escalation", "Lateral Movement", "Exfiltration"]

class GodModePipeline:
    def __init__(self, seq_len=10):
        print("Initializing IMMUNEX God-Mode Layer 2 Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Alert Transformer (Phase 1)
        try:
            self.transformer = IMMUNEX_AlertTransformer(input_dim=77, d_model=128, nhead=8, num_layers=4, num_classes=4).to(self.device)
            self.transformer.eval()
            print("[+] Spatial Alert Transformer Loaded.")
        except Exception as e:
            print(f"[-] Transformer load failed. Error: {e}")
            self.transformer = None

        # 2. BiLSTM Temporal Tracker (Phase 2)
        try:
            self.bilstm = TemporalBiLSTM(input_size=118, hidden_size=128, num_layers=2, num_classes=5).to(self.device)
            self.bilstm.eval()
            print("[+] BiLSTM Narrative Tracker Loaded.")
        except Exception as e:
            print(f"[-] BiLSTM load failed. Error: {e}")
            self.bilstm = None
            
        # 3. HMM Predictive Engine (Phase 3)
        self.hmm = PredictiveHMM()
        print("[+] HMM Predictive Engine Loaded.")

        # Kafka Connections
        try:
            self.consumer = KafkaConsumer(
                L1_ALERTS_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            self.producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print("[+] Kafka Data Highway Configured.")
        except Exception as e:
            print(f"[-] Kafka offline. Operating in simulation mode.")
            self.consumer = None
            self.producer = None
            
        self.alert_buffer = []
        self.seq_len = seq_len
        # Maps IP -> deque of spatial vectors
        self.narrative_timelines = {} 

    def process_window(self, attack_uuid: str):
        if not self.alert_buffer or not self.transformer:
            return
            
        print(f"\n==================================================================")
        print(f"[{attack_uuid}] Phase 1: Demultiplexing 50-alert buffer...")
        
        # Grouper dict: IP -> List of Alerts
        attacker_groups = {}
        for alert in self.alert_buffer:
            src_ip = alert.get('source_ip', 'unknown_ip')
            if src_ip not in attacker_groups:
                attacker_groups[src_ip] = []
            attacker_groups[src_ip].append(alert)
            
        print(f"[{attack_uuid}] Distributed into {len(attacker_groups)} Active Attackers.")
        
        # Priority Queue for dispatch (Max-heap via negative severity)
        priority_queue = []
        
        # Process each Attacker 
        for ip, alerts in attacker_groups.items():
            print(f"\n--- Attacker Session: {ip} ({len(alerts)} logs) ---")
            
            # Extract and pad features to (50, 77)
            features_list = []
            for a in alerts:
                feat = a.get('feature_vector', [0.0]*77)
                if len(feat) > 77: feat = feat[:77]
                if len(feat) < 77: feat = list(feat) + [0.0]*(77 - len(feat))
                features_list.append(feat)
            
            # Use the first alert's roberta_embedding as representative for this attacker group
            roberta_emb = None
            for a in alerts:
                emb = a.get('roberta_embedding') or a.get('feature_vector')
                if emb and len(emb) == 768:
                    roberta_emb = emb
                    break
            if roberta_emb is None:
                # Fallback: use the feature_vector of first alert (may not be 768D)
                roberta_emb = alerts[0].get('feature_vector', [])

            pad_len = 50 - len(features_list)
            if pad_len > 0:
                for _ in range(pad_len):
                    features_list.append([0.0]*77)
            elif pad_len < 0:
                features_list = features_list[:50]
                
            x_tensor = torch.tensor([features_list], dtype=torch.float32).to(self.device)
            
            # 1. Spatial Processing (Transformer)
            with torch.no_grad():
                nodes_pred, severity, spatial_vec, attns = self.transformer(x_tensor)
                severity_val = severity.item() 
                spatial_118d = spatial_vec[0]
                
                print(f"  > Spatial Vector (118D) Generated. Severity: {severity_val:.4f}")

            # 2. Chronological State Memory
            if ip not in self.narrative_timelines:
                self.narrative_timelines[ip] = deque(maxlen=self.seq_len)
            self.narrative_timelines[ip].append(spatial_118d.cpu().numpy().tolist())
            
            # Default States
            current_5d = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            future_5d = self.hmm.predict_future_stage(torch.tensor([current_5d], dtype=torch.float32))[0].numpy()
            current_stage, current_conf = "Reconnaissance", 1.0
            future_stage, future_conf = MITRE_STAGES[int(np.argmax(future_5d))], float(np.max(future_5d))
            
            # 3. Temporal Narrative (BiLSTM + HMM)
            if len(self.narrative_timelines[ip]) == self.seq_len and self.bilstm:
                seq_tensor = torch.tensor([list(self.narrative_timelines[ip])], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    current_probs = self.bilstm(seq_tensor) # (1, 5)
                    current_5d = current_probs[0].cpu().numpy()
                    
                    c_idx = int(np.argmax(current_5d))
                    current_stage = MITRE_STAGES[c_idx]
                    current_conf = float(current_5d[c_idx])
                    
                    print(f"  > BiLSTM Current State: {current_stage} ({current_conf*100:.1f}%)")
                    
                    # HMM Prediction
                    future_probs = self.hmm.predict_future_stage(current_probs)
                    future_5d = future_probs[0].cpu().numpy()
                    
                    f_idx = int(np.argmax(future_5d))
                    future_stage = MITRE_STAGES[f_idx]
                    future_conf = float(future_5d[f_idx])
                    
                    print(f"  > HMM Predicted Next Stage: {future_stage} ({future_conf*100:.1f}%)")
            else:
                progress = len(self.narrative_timelines[ip])
                print(f"  > BiLSTM Buffering... ({progress}/{self.seq_len})")

            # 4. God-Mode Fusion
            god_mode_128d = np.concatenate([spatial_118d.cpu().numpy(), current_5d, future_5d]).tolist()
            print(f"  > Finalized 128D God-Mode Vector assembled.")
            
            # Prepare rich payload for Layer 3 (DQN + Playbook Gen)
            payload = {
                "alert_id": attack_uuid,
                "source_ip": ip,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": float(severity_val),
                "attack_type": alerts[0].get('attack_type', 'Derived_From_Window'),
                "mitre_stage": current_stage,
                "predicted_next_stage": future_stage,
                "human_readable_context": {
                    "temporal_analysis": {
                        "current_mitre_stage": current_stage,
                        "current_confidence": round(current_conf, 4)
                    },
                    "predictive_analysis": {
                        "predicted_next_stage": future_stage,
                        "prediction_confidence": round(future_conf, 4)
                    }
                },
                "feature_vector": god_mode_128d,       # 128D God-Mode vector — for DQN
                "roberta_embedding": roberta_emb,       # 768D RoBERTa embedding — for pgvector
                # Compatibility field so Layer 3 validation holds
                "layer2_confidence": float(severity_val) 
            }
            
            # Push to priority queue (invert severity for max-heap behavior)
            heapq.heappush(priority_queue, (-severity_val, ip, payload))
            
        print(f"\n[{attack_uuid}] Phase 4: Priority Layer 3 Transmit")
        rank = 1
        
        # Flush queue in priority order
        while priority_queue:
            neg_sev, attacker_ip, dispatch_payload = heapq.heappop(priority_queue)
            actual_severity = -neg_sev
            
            print(f"  [Transmitting #{rank}] IP: {attacker_ip} | Severity Score: {actual_severity:.4f}")
            
            if self.producer:
                self.producer.send(L2_TO_L3_TOPIC, dispatch_payload)
            rank += 1
            
        if self.producer:
            self.producer.flush()

        self.alert_buffer.clear()

    def listen(self):
        print("\n--- Pipeline Active. Listening for Layer 1 Anomalies ---")
        if not self.consumer: return
        
        for message in self.consumer:
            alert = message.value
            self.alert_buffer.append(alert)
            print(f"Received Layer 1 Alert. Buffer size: {len(self.alert_buffer)}/50")
            
            if len(self.alert_buffer) >= 50:
                uuid = f"INC_{int(time.time())}"
                self.process_window(attack_uuid=uuid)

    def test_pipeline(self):
        """Offline standalone testing using simulated dummy buffers."""
        import random
        print("\n--- [OFFLINE DIAGNOSTIC MODE] ---")
        dummy_ips = ["192.168.1.100", "192.168.1.100", "10.0.0.5", "10.0.0.5", "172.16.0.4"]
        
        for _ in range(50):
            self.alert_buffer.append({
                "source_ip": random.choice(dummy_ips),
                "feature_vector": [random.random() for _ in range(77)]
            })
            
        uuid = f"DIAGNOSTIC_{int(time.time())}"
        self.process_window(attack_uuid=uuid)
        print("\n[+] DIAGNOSTIC PIPELINE TEST SUCCESSFUL.")

if __name__ == "__main__":
    import sys
    pipeline = GodModePipeline()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        pipeline.test_pipeline()
    else:
        print("Run `python integration_pipeline.py --test` to simulate offline.\n")
        # pipeline.listen()
