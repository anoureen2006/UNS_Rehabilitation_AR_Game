"""
train_unity.py
Training script for PPO, DQN, and A2C on UnityARRehabEnv using Stable-Baselines3.
Saves model checkpoints in `checkpoints/unity_ppo_model.zip`, `unity_dqn_model.zip`, `unity_a2c_model.zip`.
"""

import os
import time
import pandas as pd
import numpy as np

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CHECKPOINT_DIR, OUTPUT_DIR, RANDOM_SEED
from unity_env import UnityARRehabEnv
from utils import set_seeds

class UnityRehabMetricCallback(BaseCallback):
    """Callback for logging Unity trial training metrics."""
    def __init__(self, verbose=0):
        super(UnityRehabMetricCallback, self).__init__(verbose)
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
            self.episode_logs.append({
                "episode": self.episode_count,
                "reward": float(self.current_ep_reward),
                "steps": self.current_ep_steps,
                "fatigue": float(info.get("fatigue", 0.0)),
                "consecutive_timeouts": int(info.get("consecutive_timeouts", 0))
            })
            self.current_ep_reward = 0.0
            self.current_ep_steps = 0

        return True

def train_unity_agent(algo_name="PPO", total_timesteps=40000, seed=42):
    """Trains an RL model on UnityARRehabEnv."""
    print(f"\n[Unity RL Training] Training {algo_name} for {total_timesteps} steps...")
    set_seeds(seed)

    def make_env():
        return UnityARRehabEnv(seed=seed)

    env = DummyVecEnv([make_env])
    callback = UnityRehabMetricCallback()

    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
                    n_epochs=10, gamma=0.99, verbose=0, device='cpu', seed=seed)
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=20000, learning_starts=500,
                    batch_size=64, gamma=0.99, verbose=0, device='cpu', seed=seed)
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=16, gamma=0.99,
                    verbose=0, device='cpu', seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"unity_{algo_name.lower()}_model.zip")
    model.save(ckpt_path)
    print(f"Saved Unity RL model to: {ckpt_path}")
    return model, pd.DataFrame(callback.episode_logs)

if __name__ == "__main__":
    for algo in ["PPO", "DQN", "A2C"]:
        train_unity_agent(algo_name=algo, total_timesteps=30000, seed=42)
