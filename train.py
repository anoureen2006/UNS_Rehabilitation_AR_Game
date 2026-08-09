"""
train.py
Training module for PPO, DQN, and A2C algorithms using Stable-Baselines3.
Tracks rehabilitation progress metrics, saves checkpoints, and logs learning curves.
"""

import os
import time
import pandas as pd
import numpy as np
import torch

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    TOTAL_TIMESTEPS, CHECKPOINT_DIR, OUTPUT_DIR, RANDOM_SEED
)
from environment import ARRehabEnv
from utils import set_seeds
from visualization import plot_learning_curves

class RehabMetricCallback(BaseCallback):
    """
    Custom Stable-Baselines3 Callback for recording USN rehabilitation metrics:
    - Episode Reward
    - Total Room Exploration %
    - Left-Side (Neglected) Exploration %
    - Right-Side Exploration %
    - Episode Length (steps)
    - Rehabilitation Success Rate
    """
    def __init__(self, verbose=0):
        super(RehabMetricCallback, self).__init__(verbose)
        self.episode_logs = []
        self.current_ep_reward = 0.0
        self.current_ep_steps = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.current_ep_steps += 1
        reward = self.locals["rewards"][0]
        self.current_ep_reward += reward
        
        info = self.locals["infos"][0]
        dones = self.locals["dones"][0]

        if dones:
            self.episode_count += 1
            left_expl = info.get("left_expl_ratio", 0.0)
            tot_expl = info.get("total_expl_ratio", 0.0)
            right_expl = info.get("right_expl_ratio", 0.0)
            pet_found = info.get("pet_found", False)
            
            # Clinical success criterion: pet found + left exploration >= 20%
            is_success = pet_found and (left_expl >= 0.20)
            
            self.episode_logs.append({
                "episode": self.episode_count,
                "reward": float(self.current_ep_reward),
                "steps": self.current_ep_steps,
                "total_expl": float(tot_expl),
                "left_expl": float(left_expl),
                "right_expl": float(right_expl),
                "pet_found": int(pet_found),
                "success": int(is_success)
            })

            self.current_ep_reward = 0.0
            self.current_ep_steps = 0

        return True

    def get_df(self):
        """Returns recorded metrics as a Pandas DataFrame."""
        return pd.DataFrame(self.episode_logs)


def train_agent(algo_name="PPO", total_timesteps=100000, seed=42):
    """
    Trains a specified RL algorithm (PPO, DQN, or A2C) on the ARRehabEnv environment.
    
    Parameters:
        algo_name (str): 'PPO', 'DQN', or 'A2C'
        total_timesteps (int): Environment timesteps to train
        seed (int): Random seed
        
    Returns:
        model: Trained SB3 agent
        df_logs (pd.DataFrame): Training episode metrics
    """
    print(f"\n=======================================================")
    print(f"       STARTING TRAINING: {algo_name} ({total_timesteps} steps)")
    print(f"=======================================================")
    
    set_seeds(seed)
    
    # Environment creation
    def make_env():
        return ARRehabEnv(neglect_severity=0.7, seed=seed)
        
    env = DummyVecEnv([make_env])
    
    callback = RehabMetricCallback()

    # Algorithm instantiation
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            device='cpu',
            seed=seed
        )
    elif algo_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=0,
            device='cpu',
            seed=seed
        )
    elif algo_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=7e-4,
            n_steps=16,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=0,
            device='cpu',
            seed=seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Start Training
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    duration = time.time() - start_time
    
    print(f"Finished {algo_name} training in {duration:.2f} seconds ({duration/60.0:.2f} minutes).")

    # Save Checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{algo_name.lower()}_rehab_model.zip")
    model.save(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    df_logs = callback.get_df()
    return model, df_logs


def train_all_algorithms(total_timesteps=100000, seed=42):
    """Trains PPO, DQN, and A2C models and saves comparative learning curves."""
    results = {}
    models = {}

    for algo in ["PPO", "DQN", "A2C"]:
        model, df_logs = train_agent(algo_name=algo, total_timesteps=total_timesteps, seed=seed)
        results[algo] = df_logs
        models[algo] = model

    # Generate & save comparative learning curves
    plot_learning_curves(results, save_name="learning_curves_comparison.png")
    
    return models, results

if __name__ == "__main__":
    train_all_algorithms(total_timesteps=50000)
