"""
IMMUNEX HMM Predictor — Tier 4 Redesign
Replaces the hardcoded transition matrix with a learned one from train_bilstm.py.
Adds data-driven ETA estimation per stage from dwell time statistics.
"""
import numpy as np
import json
import os


class IMMUNEX_HMM_Predictor:
    def __init__(self,
                 matrix_path: str = "hmm_transition_counts.npy",
                 dwell_path:  str = "hmm_dwell_times.json",
                 adaptive_learning: bool = True):
        self.adaptive_learning = adaptive_learning
        self.stages = [
            "Reconnaissance", "Initial Access",
            "Privilege Escalation", "Lateral Movement", "Exfiltration"
        ]

        # Load learned transition counts (trained from real BiLSTM sequences)
        if os.path.exists(matrix_path):
            self.transition_counts = np.load(matrix_path).astype(np.float64)
            print(f"[HMM] Loaded learned transition matrix from {matrix_path}")
        else:
            # Fallback: uniform priors — inaccurate but won't crash
            # Run train_bilstm.py to generate the learned matrix
            self.transition_counts = np.ones((5, 5), dtype=np.float64)
            print(
                "[HMM] WARNING: No learned matrix found. "
                "Run train_bilstm.py first to generate hmm_transition_counts.npy. "
                "Using uniform priors in the meantime."
            )

        # Load dwell times for data-driven ETA estimation
        if os.path.exists(dwell_path):
            with open(dwell_path) as f:
                raw = json.load(f)
            # JSON keys are always strings — convert back to int
            self.dwell_times = {int(k): v for k, v in raw.items()}
            print(f"[HMM] Loaded dwell times from {dwell_path}")
        else:
            self.dwell_times = {}
            print(
                "[HMM] WARNING: No dwell time file found. "
                "Run train_bilstm.py first to generate hmm_dwell_times.json. "
                "ETAs will report 'unknown' until then."
            )

    @property
    def transition_matrix(self) -> np.ndarray:
        """Returns the row-normalized probability matrix from the current counts."""
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        return self.transition_counts / row_sums

    def adapt(self, prev_idx: int, curr_idx: int, weight: float = 5.0):
        """
        Real-time within-session adaptation to novel attacker behavior.
        Over-weights the observed transition so the HMM converges toward
        the actual attacker's signature during a live incident.
        """
        if self.adaptive_learning:
            self.transition_counts[prev_idx, curr_idx] += weight

    def predict_next_stage(self,
                           current_stage_probs: np.ndarray,
                           previous_stage_probs: np.ndarray = None
                           ) -> np.ndarray:
        """
        Calculates the probability distribution of the NEXT MITRE stage.
        Optionally adapts the transition matrix if a stage transition is observed.
        """
        current = np.array(current_stage_probs).flatten()

        # Normalize if not already a valid probability distribution
        if not np.isclose(current.sum(), 1.0):
            current = np.exp(current - current.max())
            current /= current.sum()

        # Adapt if a stage transition was observed since the last call
        if previous_stage_probs is not None and self.adaptive_learning:
            prev = np.array(previous_stage_probs).flatten()
            prev_idx = int(np.argmax(prev))
            curr_idx = int(np.argmax(current))
            if prev_idx != curr_idx:
                self.adapt(prev_idx, curr_idx)

        # Markov chain forward step: P(next) = P(current) · T
        future = np.dot(current, self.transition_matrix)
        return future

    def get_eta_string(self, future_stage_idx: int) -> str:
        """
        Returns a data-driven ETA string instead of the hardcoded '~2 minutes'.
        Reads mean/std dwell time for the predicted stage from the trained statistics.
        """
        info = self.dwell_times.get(future_stage_idx)
        if not info:
            return "unknown"
        mean_s = info["mean_seconds"]
        std_s  = info["std_seconds"]
        if mean_s < 60:
            return f"~{mean_s}s (±{std_s}s)"
        return f"~{mean_s // 60}m {mean_s % 60}s (±{std_s // 60}m)"

    def get_stage_name(self, index: int) -> str:
        """Returns the human-readable stage name for a given index."""
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return "Unknown"
