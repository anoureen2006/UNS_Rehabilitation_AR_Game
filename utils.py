"""
utils.py
Utility functions for seed management, metric calculations, coordinate mapping,
and baseline heuristic policies for evaluation comparison.
"""

import numpy as np
import random
import torch
from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, CELL_WIDTH, CELL_HEIGHT, NUM_CELLS

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
    
    Parameters:
        episode_logs (list of dicts): Logs containing rewards, exploration %, left expl %, step counts, etc.
        
    Returns:
        metrics (dict): Summary statistics (mean, std, success rate).
    """
    if not episode_logs:
        return {}
        
    rewards = [ep["reward"] for ep in episode_logs]
    total_expls = [ep["total_expl_ratio"] * 100.0 for ep in episode_logs]
    left_expls = [ep["left_expl_ratio"] * 100.0 for ep in episode_logs]
    right_expls = [ep["right_expl_ratio"] * 100.0 for ep in episode_logs]
    steps = [ep["step_count"] for ep in episode_logs]
    successes = [ep["success"] for ep in episode_logs]
    
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_total_expl_pct": float(np.mean(total_expls)),
        "avg_left_expl_pct": float(np.mean(left_expls)),
        "avg_right_expl_pct": float(np.mean(right_expls)),
        "avg_time_sec": float(np.mean(steps) * 0.5), # 0.5s per step
        "avg_episode_length": float(np.mean(steps)),
        "success_rate_pct": float(np.mean(successes) * 100.0),
        "asymmetry_index": float(np.mean([ep.get("expl_asymmetry", 0.0) for ep in episode_logs]))
    }

class HeuristicPolicies:
    """Baseline heuristic policies to compare against RL agents (PPO, DQN, A2C)."""
    
    @staticmethod
    def random_policy(obs=None):
        """Randomly chooses a cell in the 20x20 room."""
        return np.random.randint(0, NUM_CELLS)

    @staticmethod
    def center_policy(obs=None):
        """Always places pet in center of room (cell near (5.0, 5.0))."""
        col = 10
        row = 10
        return row * GRID_COLS + col  # cell 210

    @staticmethod
    def extreme_left_policy(obs=None):
        """Heuristically places pet in the extreme neglected left region (x ~ 1.5, y ~ 5.0)."""
        col = 3  # x ~ 1.75m
        row = np.random.randint(4, 16) # y ~ 2.0m to 8.0m
        return row * GRID_COLS + col
