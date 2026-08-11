"""
utils.py
Utility functions for seed management, metric calculations, coordinate mapping,
and evaluation policy methods (Fixed, Random, Rule-based).
"""

import numpy as np
import random
import torch
from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, NUM_CELLS
from unity_schema import DifficultyAtTrial

def set_seeds(seed=42):
    """Sets random seeds for numpy, random, and PyTorch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(episode_logs):
    """
    Aggregates performance metrics across evaluation episodes.
    Calculates: Cumulative Reward, Success Rate %, Timeout Rate %,
    Difficulty Progression, and Adaptation Stability.
    """
    if not episode_logs:
        return {}
        
    rewards = [ep["reward"] for ep in episode_logs]
    steps = [ep["step_count"] for ep in episode_logs]
    successes = [ep["success"] for ep in episode_logs]
    timeouts = [ep.get("timeout_rate", 0.0) for ep in episode_logs]
    eccentricities = [ep.get("mean_eccentricity", 10.0) for ep in episode_logs]
    ecc_stds = [ep.get("std_eccentricity", 0.0) for ep in episode_logs]
    
    return {
        "avg_cumulative_reward": float(np.mean(rewards)),
        "std_cumulative_reward": float(np.std(rewards)),
        "success_rate_pct": float(np.mean(successes) * 100.0),
        "timeout_rate_pct": float(np.mean(timeouts) * 100.0),
        "mean_eccentricity_deg": float(np.mean(eccentricities)),
        "adaptation_stability_std": float(np.mean(ecc_stds)), # Lower std = smoother adaptation
        "avg_session_steps": float(np.mean(steps))
    }

class RuleBasedPolicy:
    """
    Rule-based Heuristic Controller:
    Step-based discrete difficulty adjustment:
    - If 2 consecutive hits -> jump +5° eccentricity, +0.05 speed, -2s time limit.
    - If miss / timeout -> drop -5° eccentricity, -0.1 speed, +5s time limit.
    """
    def __init__(self):
        self.current_ecc = 10.0
        self.current_speed = 0.3
        self.current_dist = 1.0
        self.current_tlimit = 30.0
        self.recent_hits = []

    def reset(self):
        self.current_ecc = 10.0
        self.current_speed = 0.3
        self.current_dist = 1.0
        self.current_tlimit = 30.0
        self.recent_hits = []

    def get_action(self, obs):
        last_hit = bool(obs[0] > 0.5)
        self.recent_hits.append(last_hit)
        if len(self.recent_hits) > 3:
            self.recent_hits.pop(0)

        hits_count = sum(self.recent_hits)
        if hits_count >= 2:
            self.current_ecc = float(np.clip(self.current_ecc + 5.0, 5.0, 30.0))
            self.current_speed = float(np.clip(self.current_speed + 0.05, 0.2, 0.6))
            self.current_dist = float(np.clip(self.current_dist + 0.1, 0.8, 2.0))
            self.current_tlimit = float(np.clip(self.current_tlimit - 2.0, 10.0, 45.0))
        elif not last_hit:
            self.current_ecc = float(np.clip(self.current_ecc - 5.0, 5.0, 30.0))
            self.current_speed = float(np.clip(self.current_speed - 0.1, 0.2, 0.6))
            self.current_dist = float(np.clip(self.current_dist - 0.2, 0.8, 2.0))
            self.current_tlimit = float(np.clip(self.current_tlimit + 5.0, 10.0, 45.0))

        # Convert to discrete action index or difficulty
        return DifficultyAtTrial(
            speed=self.current_speed,
            eccentricity_deg=self.current_ecc,
            distance_m=self.current_dist,
            target_count=3,
            time_limit_s=self.current_tlimit
        )

class BenchmarkPolicies:
    """Evaluation Policies for Experiment 2: Fixed, Random, Rule-based."""
    
    @staticmethod
    def fixed_policy(obs=None):
        """Fixed static difficulty: Speed=0.4, Eccentricity=15°, Distance=1.2m, TimeLimit=30s."""
        return DifficultyAtTrial(speed=0.4, eccentricity_deg=15.0, distance_m=1.2, target_count=3, time_limit_s=30.0)

    @staticmethod
    def random_policy(obs=None):
        """Random unadapted parameter choice per trial."""
        speed = random.choice([0.2, 0.3, 0.4, 0.5, 0.6])
        ecc = random.choice([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        dist = random.choice([0.8, 1.0, 1.2, 1.5, 2.0])
        tlimit = random.choice([10.0, 15.0, 20.0, 30.0, 45.0])
        return DifficultyAtTrial(speed=speed, eccentricity_deg=ecc, distance_m=dist, target_count=3, time_limit_s=tlimit)
