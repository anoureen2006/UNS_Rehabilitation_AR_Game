"""
evaluate.py
Experiment 2 — RL Validation & Comparative Benchmark Evaluation Script.
Evaluates simulated patients across 6 methods:
1. Fixed (Adaptive: No)
2. Random (Adaptive: No)
3. Rule-based (Adaptive: Yes)
4. PPO (Adaptive: Yes)
5. DQN (Adaptive: Yes)
6. A2C (Adaptive: Yes)

Measures: Cumulative Reward, Success Rate %, Timeout Rate %, Difficulty Progression, and Adaptation Stability.
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C

from config import CHECKPOINT_DIR, OUTPUT_DIR, EVAL_EPISODES, RANDOM_SEED
from unity_env import UnityARRehabEnv
from utils import set_seeds, calculate_metrics, BenchmarkPolicies, RuleBasedPolicy
from visualization import (
    plot_experiment2_benchmark,
    plot_difficulty_progression_comparison
)

def evaluate_unity_policy(env, policy_func_or_model, is_sb3=False, is_rule_based=False, num_episodes=50, seed=42):
    """
    Evaluates a policy on UnityARRehabEnv across num_episodes.
    
    Returns:
        episode_logs (list of dicts): Summary metrics per episode
        progression_data (dict): Trial-by-trial difficulty parameters for the final episode
    """
    set_seeds(seed)
    episode_logs = []
    final_progression = {"eccentricity": [], "speed": [], "distance": [], "time_limit": []}

    rule_policy = RuleBasedPolicy() if is_rule_based else None

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        if is_rule_based:
            rule_policy.reset()

        done = False
        step_count = 0
        ep_reward = 0.0
        hits = 0
        timeouts = 0
        ep_eccentricities = []

        while not done:
            if is_sb3:
                action, _ = policy_func_or_model.predict(obs, deterministic=True)
            elif is_rule_based:
                diff_obj = rule_policy.get_action(obs)
                # Map diff_obj to environment step
                action = env.action_space.sample() # internal step accepts discrete action or direct override
            else:
                diff_obj = policy_func_or_model(obs)
                action = env.action_space.sample()

            obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Override diff for non-SB3 benchmark policies
            if not is_sb3:
                trial_diff = diff_obj
            else:
                trial_diff = step_info["trial"].difficulty_at_trial

            ep_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_info["trial"].hit:
                hits += 1
            if step_info["trial"].reaction_time_ms >= (trial_diff.time_limit_s * 1000.0):
                timeouts += 1

            ep_eccentricities.append(trial_diff.eccentricity_deg)

            if ep == num_episodes - 1:
                final_progression["eccentricity"].append(trial_diff.eccentricity_deg)
                final_progression["speed"].append(trial_diff.speed)
                final_progression["distance"].append(trial_diff.distance_m)
                final_progression["time_limit"].append(trial_diff.time_limit_s)

        success = (hits / max(1, step_count)) >= 0.5
        timeout_rate = timeouts / max(1, step_count)

        episode_logs.append({
            "episode": ep + 1,
            "reward": float(ep_reward),
            "step_count": step_count,
            "success": int(success),
            "timeout_rate": float(timeout_rate),
            "mean_eccentricity": float(np.mean(ep_eccentricities)),
            "std_eccentricity": float(np.std(ep_eccentricities))
        })

    return episode_logs, final_progression


def run_experiment2_validation(num_episodes=50, seed=42):
    """
    Executes Experiment 2 — RL Validation & Comparative Benchmark Study.
    Compares: Fixed, Random, Rule-based, PPO, DQN, A2C.
    """
    print(f"\n=======================================================")
    print(f"    EXPERIMENT 2 — RL VALIDATION BENCHMARK ({num_episodes} episodes)")
    print(f"=======================================================")

    env = UnityARRehabEnv(seed=seed)
    results_summary = []
    progression_dict = {}

    # 1. Method: Fixed (Adaptive: No)
    print("Evaluating Method 1/6: Fixed Difficulty (Adaptive: No)...")
    logs, prog = evaluate_unity_policy(env, BenchmarkPolicies.fixed_policy, is_sb3=False, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Fixed"
    m["Adaptive"] = "No"
    results_summary.append(m)
    progression_dict["Fixed"] = prog

    # 2. Method: Random (Adaptive: No)
    print("Evaluating Method 2/6: Random Difficulty (Adaptive: No)...")
    logs, prog = evaluate_unity_policy(env, BenchmarkPolicies.random_policy, is_sb3=False, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Random"
    m["Adaptive"] = "No"
    results_summary.append(m)
    progression_dict["Random"] = prog

    # 3. Method: Rule-based (Adaptive: Yes)
    print("Evaluating Method 3/6: Rule-based Heuristic (Adaptive: Yes)...")
    logs, prog = evaluate_unity_policy(env, None, is_sb3=False, is_rule_based=True, num_episodes=num_episodes, seed=seed)
    m = calculate_metrics(logs)
    m["Method"] = "Rule-based"
    m["Adaptive"] = "Yes"
    results_summary.append(m)
    progression_dict["Rule-based"] = prog

    # 4-6. RL Methods: PPO, DQN, A2C (Adaptive: Yes)
    for algo in ["PPO", "DQN", "A2C"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"unity_{algo.lower()}_model.zip")
        if os.path.exists(ckpt_path):
            print(f"Evaluating Method: {algo} Deep RL Agent (Adaptive: Yes)...")
            if algo == "PPO":
                model = PPO.load(ckpt_path, device='cpu')
            elif algo == "DQN":
                model = DQN.load(ckpt_path, device='cpu')
            else:
                model = A2C.load(ckpt_path, device='cpu')

            logs, prog = evaluate_unity_policy(env, model, is_sb3=True, num_episodes=num_episodes, seed=seed)
            m = calculate_metrics(logs)
            m["Method"] = algo
            m["Adaptive"] = "Yes"
            results_summary.append(m)
            progression_dict[algo] = prog
        else:
            print(f"Warning: Model checkpoint for {algo} not found at {ckpt_path}. Skipping.")

    summary_df = pd.DataFrame(results_summary)
    
    # Save CSV Results
    csv_path = os.path.join(OUTPUT_DIR, "exp2_benchmark_results.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n=======================================================")
    print("       EXPERIMENT 2 — RL VALIDATION RESULTS SUMMARY     ")
    print("=======================================================")
    cols_to_print = ['Method', 'Adaptive', 'avg_cumulative_reward', 'success_rate_pct', 'timeout_rate_pct', 'adaptation_stability_std']
    print(summary_df[cols_to_print].to_string(index=False))
    print("=======================================================\n")

    # Generate Publication Figures
    plot_experiment2_benchmark(summary_df, save_name="exp2_benchmark_comparison.png")
    plot_difficulty_progression_comparison(progression_dict, save_name="exp2_difficulty_progression.png")

    return summary_df, progression_dict

if __name__ == "__main__":
    run_experiment2_validation(num_episodes=EVAL_EPISODES)
