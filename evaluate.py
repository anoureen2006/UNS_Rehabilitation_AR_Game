"""
evaluate.py
Evaluation and comparative benchmark script for AR Rehabilitation Digital Pet Placement.
Evaluates baseline heuristic policies (Random, Center, Extreme Left) against trained PPO, DQN, A2C models.
Computes research metrics (Average Reward, Exploration %, Left Exploration %, Time to Pet, Success Rate).
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C

from config import CHECKPOINT_DIR, OUTPUT_DIR, EVAL_EPISODES, RANDOM_SEED
from environment import ARRehabEnv
from utils import set_seeds, calculate_metrics, HeuristicPolicies
from visualization import plot_evaluation_comparison, plot_room_and_trajectory, plot_exploration_heatmap, create_interactive_trajectory_plotly

def evaluate_policy(env, policy_func_or_model, is_sb3_model=False, num_episodes=50, seed=42):
    """
    Runs evaluation episodes for a given policy or model.
    
    Parameters:
        env (ARRehabEnv): Gymnasium rehabilitation environment
        policy_func_or_model: Callable policy function or SB3 model instance
        is_sb3_model (bool): True if policy is a Stable-Baselines3 model
        num_episodes (int): Number of test episodes to run
        seed (int): Base random seed
        
    Returns:
        episode_logs (list of dicts): Episode summary records
        last_trajectory_data (dict): Data for rendering the final test episode
    """
    set_seeds(seed)
    episode_logs = []
    last_data = {}

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step_count = 0
        ep_reward = 0.0
        
        # Step 0: Policy selects pet placement location
        if is_sb3_model:
            action, _ = policy_func_or_model.predict(obs, deterministic=True)
        else:
            action = policy_func_or_model(obs)

        while not done:
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

        left_expl = step_info.get("left_expl_ratio", 0.0)
        tot_expl = step_info.get("total_expl_ratio", 0.0)
        right_expl = step_info.get("right_expl_ratio", 0.0)
        pet_found = step_info.get("pet_found", False)
        
        # Clinical success: Pet found after exploring at least 20% of neglected left side
        success = pet_found and (left_expl >= 0.20)

        episode_logs.append({
            "episode": ep + 1,
            "reward": float(ep_reward),
            "step_count": step_count,
            "total_expl_ratio": float(tot_expl),
            "left_expl_ratio": float(left_expl),
            "right_expl_ratio": float(right_expl),
            "pet_found": int(pet_found),
            "success": int(success),
            "expl_asymmetry": step_info.get("expl_asymmetry", 0.0)
        })

        if ep == num_episodes - 1:
            last_data = {
                "patient": env.patient,
                "pet": env.pet,
                "env": env,
                "visited_grid": env.patient.visited_grid.copy()
            }

    return episode_logs, last_data


def run_benchmark(num_episodes=50, seed=42):
    """
    Executes full evaluation comparison across baseline heuristics and trained RL agents.
    """
    print(f"\n=======================================================")
    print(f"       RUNNING BENCHMARK EVALUATION ({num_episodes} episodes per policy)")
    print(f"=======================================================")

    env = ARRehabEnv(neglect_severity=0.7, seed=seed)
    results_summary = []

    # 1. Baseline: Random Placement
    print("Evaluating Policy 1/6: Baseline - Random Placement...")
    logs, _ = evaluate_policy(env, HeuristicPolicies.random_policy, is_sb3_model=False, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Random"
    results_summary.append(m)

    # 2. Baseline: Center Placement
    print("Evaluating Policy 2/6: Baseline - Center Placement...")
    logs, _ = evaluate_policy(env, HeuristicPolicies.center_policy, is_sb3_model=False, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Center"
    results_summary.append(m)

    # 3. Baseline: Extreme Left Placement
    print("Evaluating Policy 3/6: Baseline - Extreme Left Placement...")
    logs, _ = evaluate_policy(env, HeuristicPolicies.extreme_left_policy, is_sb3_model=False, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Extreme Left"
    results_summary.append(m)

    # 4. RL Agents (PPO, DQN, A2C)
    for algo in ["PPO", "DQN", "A2C"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{algo.lower()}_rehab_model.zip")
        if os.path.exists(ckpt_path):
            print(f"Evaluating Policy: Trained {algo} Agent...")
            if algo == "PPO":
                model = PPO.load(ckpt_path)
            elif algo == "DQN":
                model = DQN.load(ckpt_path)
            else:
                model = A2C.load(ckpt_path)

            logs, last_data = evaluate_policy(env, model, is_sb3_model=True, num_episodes=num_episodes, seed=seed)
            m = calculate_metrics(logs)
            m["Method"] = algo
            results_summary.append(m)

            # Generate sample visualizations for best performing RL policy (e.g. PPO)
            if algo == "PPO" and last_data:
                plot_room_and_trajectory(last_data["env"], last_data["patient"], last_data["pet"],
                                         title=f"PPO Policy - Patient Trajectory",
                                         save_name="ppo_sample_trajectory.png")
                plot_exploration_heatmap(last_data["visited_grid"],
                                        title="PPO Policy - Spatial Exploration Heatmap",
                                        save_name="ppo_exploration_heatmap.png")
                create_interactive_trajectory_plotly(last_data["patient"].trajectory, last_data["pet"].pos,
                                                     last_data["env"].obstacles,
                                                     save_name="ppo_interactive_trajectory.html")
        else:
            print(f"Warning: Checkpoint for {algo} not found at {ckpt_path}. Skipping.")

    summary_df = pd.DataFrame(results_summary)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_benchmark_results.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n=======================================================")
    print("                 BENCHMARK RESULTS SUMMARY              ")
    print("=======================================================")
    print(summary_df[['Method', 'avg_reward', 'avg_total_expl_pct', 'avg_left_expl_pct', 'avg_time_sec', 'success_rate_pct']].to_string(index=False))
    print("=======================================================\n")

    # Generate comparison plot
    plot_evaluation_comparison(summary_df, save_name="eval_benchmark_comparison.png")
    return summary_df

if __name__ == "__main__":
    run_benchmark(num_episodes=EVAL_EPISODES)
