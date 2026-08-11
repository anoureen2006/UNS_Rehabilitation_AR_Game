"""
utils.py
Utility functions for multi-seed statistical evaluation, scientific metrics computation,
coordinate mapping, and benchmark policy methods (Fixed, Random, Rule-based).
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

def compute_adaptation_smoothness(eccentricities, e_min=5.0, e_max=35.0):
    """
    Computes Adaptation Smoothness (AS):
    AS = 1 - (1 / (T - 1)) * sum(|e_t - e_{t-1}| / (e_max - e_min))
    Returns 1.0 for perfectly smooth adaptation, lower for erratic staircasing.
    """
    if len(eccentricities) < 2:
        return 1.0
    
    diffs = np.abs(np.diff(eccentricities))
    range_e = max(1.0, e_max - e_min)
    normalized_changes = diffs / range_e
    mean_change = np.mean(normalized_changes)
    
    return float(np.clip(1.0 - mean_change, 0.0, 1.0))

def calculate_multiseed_metrics(all_seed_results):
    """
    Computes multi-seed statistics (Mean, SD, 95% Confidence Interval) across independent runs.
    
    Parameters:
        all_seed_results (list of dicts): Logs across independent random seeds.
        
    Returns:
        metrics (dict): Scientific metrics for publication tables.
    """
    if not all_seed_results:
        return {}
        
    rewards = [res["avg_reward"] for res in all_seed_results]
    successes = [res["success_rate_pct"] for res in all_seed_results]
    timeouts = [res["timeout_rate_pct"] for res in all_seed_results]
    max_eccs = [res["max_eccentricity_deg"] for res in all_seed_results]
    delta_eccs = [res["delta_eccentricity_deg"] for res in all_seed_results]
    smoothness_scores = [res["adaptation_smoothness"] for res in all_seed_results]

    N = len(all_seed_results)
    
    def mean_sd_ci(data):
        mean_val = float(np.mean(data))
        sd_val = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        ci_val = float(1.96 * sd_val / np.sqrt(N)) if len(data) > 1 else 0.0
        return mean_val, sd_val, ci_val

    r_mean, r_sd, r_ci = mean_sd_ci(rewards)
    s_mean, s_sd, s_ci = mean_sd_ci(successes)
    t_mean, t_sd, t_ci = mean_sd_ci(timeouts)
    e_mean, e_sd, e_ci = mean_sd_ci(max_eccs)
    de_mean, de_sd, de_ci = mean_sd_ci(delta_eccs)
    as_mean, as_sd, as_ci = mean_sd_ci(smoothness_scores)

    return {
        "mean_reward": r_mean,
        "sd_reward": r_sd,
        "ci95_reward": r_ci,
        "reward_str": f"{r_mean:.1f} ± {r_sd:.1f}",
        "success_rate_pct": s_mean,
        "timeout_rate_pct": t_mean,
        "max_eccentricity_deg": e_mean,
        "delta_eccentricity_deg": de_mean,
        "adaptation_smoothness": as_mean,
        "smoothness_str": f"{as_mean:.3f} ± {as_sd:.3f}"
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
