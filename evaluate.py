"""
evaluate.py
Experiment 2 — Comparative Evaluation of Adaptive Difficulty Policies.
Executes multi-seed evaluation across 5 independent random seeds per method:
1. PPO (Adaptive: Yes)
2. A2C (Adaptive: Yes)
3. DQN (Adaptive: Yes)
4. Rule-based (Adaptive: Yes)
5. Random (Adaptive: No)
6. Fixed (Adaptive: No)

Evaluates: Cumulative Reward (Mean ± SD), Success Rate %, Timeout Rate %,
Maximum Neglected Eccentricity (E_max), Difficulty Progression (ΔE), and Adaptation Smoothness (AS).
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C

from config import CHECKPOINT_DIR, OUTPUT_DIR, EVAL_EPISODES, RANDOM_SEED
from unity_env import UnityARRehabEnv
from utils import set_seeds, calculate_multiseed_metrics, compute_adaptation_smoothness, BenchmarkPolicies, RuleBasedPolicy
from visualization import (
    plot_experiment2_benchmark,
    plot_difficulty_progression_comparison
)

EVAL_SEEDS = [42, 101, 202, 303, 404]  # 5 independent random seeds

def evaluate_single_seed(env, policy_func_or_model, is_sb3=False, is_rule_based=False, num_episodes=20, seed=42):
    """Evaluates a policy on a single random seed across num_episodes."""
    set_seeds(seed)
    ep_rewards = []
    ep_successes = []
    ep_timeouts = []
    ep_max_eccs = []
    ep_delta_eccs = []
    ep_smoothness = []
    
    progression_sample = {"eccentricity": [], "speed": [], "distance": [], "time_limit": []}
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
        eccentricities = []

        while not done:
            if is_sb3:
                action, _ = policy_func_or_model.predict(obs, deterministic=True)
                diff_obj = None
            elif is_rule_based:
                diff_obj = rule_policy.get_action(obs)
                action = env.action_space.sample()
            else:
                diff_obj = policy_func_or_model(obs)
                action = env.action_space.sample()

            obs, reward, terminated, truncated, step_info = env.step(action)
            
            trial_diff = diff_obj if not is_sb3 else step_info["trial"].difficulty_at_trial
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_info["trial"].hit:
                hits += 1
            if step_info["trial"].reaction_time_ms >= (trial_diff.time_limit_s * 1000.0 - 100.0):
                timeouts += 1

            eccentricities.append(trial_diff.eccentricity_deg)

            if ep == num_episodes - 1:
                progression_sample["eccentricity"].append(trial_diff.eccentricity_deg)
                progression_sample["speed"].append(trial_diff.speed)
                progression_sample["distance"].append(trial_diff.distance_m)
                progression_sample["time_limit"].append(trial_diff.time_limit_s)

        ep_rewards.append(ep_reward)
        ep_successes.append(hits / max(1, step_count))
        ep_timeouts.append(timeouts / max(1, step_count))
        
        max_ecc = float(np.max(eccentricities)) if eccentricities else 10.0
        delta_ecc = float(eccentricities[-1] - eccentricities[0]) if len(eccentricities) > 1 else 0.0
        as_score = compute_adaptation_smoothness(eccentricities)
        
        ep_max_eccs.append(max_ecc)
        ep_delta_eccs.append(delta_ecc)
        ep_smoothness.append(as_score)

    return {
        "avg_reward": float(np.mean(ep_rewards)),
        "success_rate_pct": float(np.mean(ep_successes) * 100.0),
        "timeout_rate_pct": float(np.mean(ep_timeouts) * 100.0),
        "max_eccentricity_deg": float(np.mean(ep_max_eccs)),
        "delta_eccentricity_deg": float(np.mean(ep_delta_eccs)),
        "adaptation_smoothness": float(np.mean(ep_smoothness))
    }, progression_sample


def run_experiment2_validation(seeds=EVAL_SEEDS, num_episodes=20):
    """
    Executes Experiment 2 — Comparative Evaluation of Adaptive Difficulty Policies across 5 independent seeds.
    Compares: PPO, A2C, DQN, Rule-based, Random, Fixed.
    """
    print(f"\n=========================================================================")
    print(f" EXPERIMENT 2 — COMPARATIVE EVALUATION ({len(seeds)} SEEDS x {num_episodes} EPISODES)")
    print(f"=========================================================================")

    env = UnityARRehabEnv(seed=RANDOM_SEED)
    summary_list = []
    progression_dict = {}

    methods_config = [
        ("PPO", "Yes", True, False, "PPO"),
        ("A2C", "Yes", True, False, "A2C"),
        ("DQN", "Yes", True, False, "DQN"),
        ("Rule-based", "Yes", False, True, None),
        ("Random", "No", False, False, BenchmarkPolicies.random_policy),
        ("Fixed", "No", False, False, BenchmarkPolicies.fixed_policy)
    ]

    for name, adaptive, is_sb3, is_rule_based, policy_target in methods_config:
        print(f"Evaluating Method: {name} (Adaptive: {adaptive}) across {len(seeds)} independent seeds...")
        seed_results = []
        sample_progression = None

        model = None
        if is_sb3:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"unity_{policy_target.lower()}_model.zip")
            if os.path.exists(ckpt_path):
                if policy_target == "PPO":
                    model = PPO.load(ckpt_path, device='cpu')
                elif policy_target == "DQN":
                    model = DQN.load(ckpt_path, device='cpu')
                else:
                    model = A2C.load(ckpt_path, device='cpu')
            else:
                print(f"Warning: Checkpoint for {policy_target} not found at {ckpt_path}. Skipping.")
                continue

        for s in seeds:
            target_obj = model if is_sb3 else policy_target
            m_seed, prog = evaluate_single_seed(
                env, target_obj, is_sb3=is_sb3, is_rule_based=is_rule_based,
                num_episodes=num_episodes, seed=s
            )
            seed_results.append(m_seed)
            if sample_progression is None:
                sample_progression = prog

        metrics = calculate_multiseed_metrics(seed_results)
        metrics["Method"] = name
        metrics["Adaptive"] = adaptive
        summary_list.append(metrics)
        progression_dict[name] = sample_progression

    summary_df = pd.DataFrame(summary_list)
    
    # Save CSV Results
    csv_path = os.path.join(OUTPUT_DIR, "exp2_benchmark_results.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n=========================================================================================================")
    print("                      EXPERIMENT 2 — FINAL COMPARATIVE BENCHMARK TABLE                                 ")
    print("=========================================================================================================")
    cols_to_print = ['Method', 'Adaptive', 'reward_str', 'success_rate_pct', 'timeout_rate_pct', 'max_eccentricity_deg', 'delta_eccentricity_deg', 'smoothness_str']
    print(summary_df[cols_to_print].to_string(index=False))
    print("=========================================================================================================\n")

    # Generate Publication Figures
    plot_experiment2_benchmark(summary_df, save_name="exp2_benchmark_comparison.png")
    plot_difficulty_progression_comparison(progression_dict, save_name="exp2_difficulty_progression.png")

    return summary_df, progression_dict

if __name__ == "__main__":
    run_experiment2_validation(seeds=EVAL_SEEDS, num_episodes=20)
