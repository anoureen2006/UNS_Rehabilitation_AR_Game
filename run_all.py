"""
run_all.py
Master execution script for the AR Rehabilitation RL Simulation project.
Runs training across PPO, DQN, and A2C, evaluates all trained policies against baselines,
and generates all publication-quality research figures.
"""

import time
import os
from train import train_all_algorithms
from evaluate import run_benchmark
from config import TOTAL_TIMESTEPS, OUTPUT_DIR

def main():
    print("\n" + "="*70)
    print("      AR REHABILITATION DIGITAL PET RL SIMULATION PIPELINE")
    print("      Unilateral Spatial Neglect (USN) Target Placement")
    print("="*70 + "\n")

    start_time = time.time()
    
    # Step 1: Train PPO, DQN, A2C agents
    # Use 30,000 steps per agent for fast complete pipeline execution (or 100,000+ for full convergence)
    train_timesteps = 30000 
    print(f"[Phase 1/2] Training RL Agents (PPO, DQN, A2C) for {train_timesteps} steps each...")
    models, train_results = train_all_algorithms(total_timesteps=train_timesteps, seed=42)

    # Step 2: Evaluate Baselines vs Trained Agents
    print(f"\n[Phase 2/2] Running Benchmark Evaluation across 50 episodes per policy...")
    eval_df = run_benchmark(num_episodes=50, seed=42)

    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed:.2f} seconds ({elapsed/60.0:.2f} minutes).")
    print(f"All output visualizations and evaluation summaries saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
