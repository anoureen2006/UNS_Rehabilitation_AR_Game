"""
visualization.py
Visualization suite for AR Rehabilitation Digital Pet RL Simulation & Experiment 2 Validation.
Generates research publication plots:
1. Experiment 2 Method Comparison (Fixed vs Random vs Rule-based vs PPO vs DQN vs A2C)
2. Difficulty Progression Trajectory over Trials (Rule-based staircasing vs PPO smooth adaptation)
3. 3-RL Algorithm Comparison Learning Curves (PPO vs DQN vs A2C)
4. Exploration Heatmaps and Trajectory Plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, LEFT_BOUND, EXTREME_LEFT_BOUND, OUTPUT_DIR

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def plot_experiment2_benchmark(df_summary, save_name="exp2_benchmark_comparison.png"):
    """
    Renders 4-panel comparison bar charts for Experiment 2:
    Method vs Cumulative Reward, Success Rate %, Timeout Rate %, Adaptation Stability.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    methods = df_summary['Method']
    palette = ['#e41a1c', '#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#a65628']

    # 1. Cumulative Reward
    axes[0, 0].bar(methods, df_summary['avg_cumulative_reward'], color=palette)
    axes[0, 0].set_title("Cumulative Reward (Higher is Better)", weight='bold', fontsize=12)
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].tick_params(axis='x', rotation=25)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Success Rate %
    axes[0, 1].bar(methods, df_summary['success_rate_pct'], color=palette)
    axes[0, 1].set_title("Success Rate % (Higher is Better)", weight='bold', fontsize=12)
    axes[0, 1].set_ylabel("Success %")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].tick_params(axis='x', rotation=25)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Timeout Rate %
    axes[1, 0].bar(methods, df_summary['timeout_rate_pct'], color=palette)
    axes[1, 0].set_title("Timeout Rate % (Lower is Better)", weight='bold', fontsize=12)
    axes[1, 0].set_ylabel("Timeout %")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].tick_params(axis='x', rotation=25)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 4. Adaptation Stability (Std Dev)
    axes[1, 1].bar(methods, df_summary['adaptation_stability_std'], color=palette)
    axes[1, 1].set_title("Adaptation Instability / Variance (Lower is Smoother)", weight='bold', fontsize=12)
    axes[1, 1].set_ylabel("Eccentricity Std Dev (°)")
    axes[1, 1].tick_params(axis='x', rotation=25)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Experiment 2 — RL Validation & Benchmark Comparison", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_difficulty_progression_comparison(progression_dict, save_name="exp2_difficulty_progression.png"):
    """
    Plots trial-by-trial difficulty progression comparing Rule-based (oscillatory staircasing)
    vs PPO (smooth continuous adaptation).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    
    colors = {
        'Fixed': '#e41a1c',
        'Random': '#ff7f00',
        'Rule-based': '#377eb8',
        'PPO': '#4daf4a',
        'DQN': '#984ea3',
        'A2C': '#a65628'
    }

    for method, data in progression_dict.items():
        c = colors.get(method, 'black')
        trials = range(1, len(data['eccentricity']) + 1)
        
        # 1. Eccentricity Progression
        axes[0].plot(trials, data['eccentricity'], label=method, color=c, linewidth=2.0,
                     linestyle='--' if method == 'Rule-based' else '-')
        
        # 2. Speed Progression
        axes[1].plot(trials, data['speed'], label=method, color=c, linewidth=2.0,
                     linestyle='--' if method == 'Rule-based' else '-')

    axes[0].set_title("Target Eccentricity (°)", weight='bold', fontsize=12)
    axes[0].set_ylabel("Eccentricity (°)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].set_title("Target Speed (m/s)", weight='bold', fontsize=12)
    axes[1].set_xlabel("Trial Number")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Experiment 2 — Trial Difficulty Progression & Adaptation Stability", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_learning_curves(results_dict, save_name="learning_curves_comparison.png"):
    """Plots comparative learning curves for 3 RL algorithms (PPO vs DQN vs A2C)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    colors = {'PPO': '#4daf4a', 'DQN': '#984ea3', 'A2C': '#a65628'}

    for name, df in results_dict.items():
        c = colors.get(name, 'black')
        window = max(1, len(df) // 20)
        
        # 1. Reward
        smooth_r = df['reward'].rolling(window=window, min_periods=1).mean()
        axes[0, 0].plot(df['episode'], smooth_r, label=name, color=c, linewidth=2)
        axes[0, 0].set_title("Reward Progression", weight='bold')
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].grid(True)

        # 2. Steps / Trial Duration
        smooth_steps = df['steps'].rolling(window=window, min_periods=1).mean()
        axes[0, 1].plot(df['episode'], smooth_steps, label=name, color=c, linewidth=2)
        axes[0, 1].set_title("Session Trial Steps", weight='bold')
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True)

        # 3. Fatigue Accumulation
        smooth_fatigue = df.get('fatigue', pd.Series([0]*len(df))).rolling(window=window, min_periods=1).mean()
        axes[1, 0].plot(df['episode'], smooth_fatigue, label=name, color=c, linewidth=2)
        axes[1, 0].set_title("Fatigue Build-Up Rate", weight='bold')
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Fatigue Factor")
        axes[1, 0].grid(True)

        # 4. Timeout Rate
        smooth_timeouts = df.get('consecutive_timeouts', pd.Series([0]*len(df))).rolling(window=window, min_periods=1).mean()
        axes[1, 1].plot(df['episode'], smooth_timeouts, label=name, color=c, linewidth=2)
        axes[1, 1].set_title("Consecutive Timeout Failure Rate", weight='bold')
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Timeout Count")
        axes[1, 1].grid(True)

    for ax in axes.flat:
        ax.legend(loc='upper right')

    plt.suptitle("3-RL Algorithm Comparison (PPO vs DQN vs A2C)", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_room_and_trajectory(env, patient, pet, title="AR Rehab Simulation Trajectory", save_name="room_trajectory.png"):
    """Renders 2D room layout with patient path, FOV cone, and pet target."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    ax.add_patch(patches.Rectangle((0, 0), EXTREME_LEFT_BOUND, ROOM_HEIGHT, color='#ffcccc', alpha=0.4, label='Severe Neglected Region'))
    ax.add_patch(patches.Rectangle((EXTREME_LEFT_BOUND, 0), LEFT_BOUND - EXTREME_LEFT_BOUND, ROOM_HEIGHT, color='#ffe6cc', alpha=0.3, label='Neglected Region'))
    ax.add_patch(patches.Rectangle((LEFT_BOUND, 0), ROOM_WIDTH - LEFT_BOUND, ROOM_HEIGHT, color='#e6f2ff', alpha=0.2, label='Non-Neglected Region'))

    traj = np.array(patient.trajectory)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], color='#800080', linewidth=2.0, label='Patient Path')
        ax.scatter(traj[0, 0], traj[0, 1], color='blue', s=80, label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=100, label='Final Pos')

    ax.scatter(pet.pos[0], pet.pos[1], color='#FF1493', s=200, marker='*', zorder=7, label='Digital Pet')
    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(0, ROOM_HEIGHT)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_exploration_heatmap(visited_grid, title="Spatial Exploration Heatmap", save_name="exploration_heatmap.png"):
    """Renders 20x20 spatial heatmap of patient room exploration."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(visited_grid, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Visit Count'}, ax=ax)
    ax.set_title(title, fontsize=12, weight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def create_interactive_trajectory_plotly(patient_trajectory, pet_pos, obstacles, save_name="interactive_trajectory.html"):
    """Generates interactive Plotly HTML viewer for patient trajectory."""
    traj = np.array(patient_trajectory)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers',
                             line=dict(color='purple', width=3), name='Patient Path'))
    fig.add_trace(go.Scatter(x=[pet_pos[0]], y=[pet_pos[1]], mode='markers',
                             marker=dict(size=18, color='DeepPink', symbol='star'), name='Digital Pet'))

    fig.update_layout(title="Interactive Patient Trajectory", width=700, height=700)
    save_path = os.path.join(OUTPUT_DIR, save_name)
    fig.write_html(save_path)
    return save_path
