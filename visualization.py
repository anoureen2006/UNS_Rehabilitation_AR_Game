"""
visualization.py
Visualization suite for AR Rehabilitation Digital Pet RL Simulation.
Includes Matplotlib 2D room layouts, exploration heatmaps, learning curve plots,
and Plotly interactive 3D surface/trajectory visualizations.
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

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def plot_room_and_trajectory(env, patient, pet, title="AR Rehab Simulation Trajectory", save_name="room_trajectory.png"):
    """
    Renders 2D room layout with procedural obstacles, patient trajectory, FOV cone, and digital pet position.
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    
    # 1. Draw Left / Right Neglect Boundary Zones
    ax.add_patch(patches.Rectangle((0, 0), EXTREME_LEFT_BOUND, ROOM_HEIGHT, color='#ffcccc', alpha=0.4, label='Severe Neglected Region (x < 2.5m)'))
    ax.add_patch(patches.Rectangle((EXTREME_LEFT_BOUND, 0), LEFT_BOUND - EXTREME_LEFT_BOUND, ROOM_HEIGHT, color='#ffe6cc', alpha=0.3, label='Neglected Region (x < 5.0m)'))
    ax.add_patch(patches.Rectangle((LEFT_BOUND, 0), ROOM_WIDTH - LEFT_BOUND, ROOM_HEIGHT, color='#e6f2ff', alpha=0.2, label='Non-Neglected Region (x >= 5.0m)'))

    # Draw Grid Lines
    for c in range(GRID_COLS + 1):
        ax.axvline(c * (ROOM_WIDTH / GRID_COLS), color='gray', linestyle=':', alpha=0.3)
    for r in range(GRID_ROWS + 1):
        ax.axhline(r * (ROOM_HEIGHT / GRID_ROWS), color='gray', linestyle=':', alpha=0.3)

    # 2. Draw Room Obstacles
    obs_colors = {
        'chair': '#8B4513', 'table': '#A0522D', 'plant': '#228B22',
        'book': '#4682B4', 'cup': '#DAA520', 'door': '#800000', 'window': '#1E90FF'
    }
    for obs in env.obstacles:
        c = obs_colors.get(obs['type'], 'black')
        circle = plt.Circle(obs['pos'], obs['radius'], color=c, alpha=0.7)
        ax.add_patch(circle)
        ax.text(obs['pos'][0], obs['pos'][1], obs['type'][0].upper(), color='white',
                fontsize=8, weight='bold', ha='center', va='center')

    # 3. Draw Patient Trajectory
    traj = np.array(patient.trajectory)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], color='#800080', linewidth=2.0, linestyle='-', label='Patient Path')
        ax.scatter(traj[0, 0], traj[0, 1], color='blue', s=80, zorder=5, label='Start Position')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=100, zorder=5, label='Final Position')

        # Draw Heading Arrow & Visual FOV Cone
        last_pos = traj[-1]
        heading = patient.heading
        fov_rad = patient.fov_rad
        view_dist = patient.max_view_dist
        
        # Heading Vector
        ax.arrow(last_pos[0], last_pos[1], 0.8 * np.cos(heading), 0.8 * np.sin(heading),
                 head_width=0.25, head_length=0.25, fc='purple', ec='purple', zorder=6)

        # FOV Wedge
        wedge_start = np.degrees(heading - fov_rad / 2)
        wedge_end = np.degrees(heading + fov_rad / 2)
        fov_wedge = patches.Wedge(last_pos, view_dist, wedge_start, wedge_end,
                                  color='purple', alpha=0.15, label='Patient FOV Cone (120°)')
        ax.add_patch(fov_wedge)

    # 4. Draw Digital Pet Position
    ax.scatter(pet.pos[0], pet.pos[1], color='#FF1493', s=200, marker='*', zorder=7, label='Digital Pet 🐶')
    pet_reach = plt.Circle(pet.pos, patient.reach_radius, color='#FF1493', fill=False, linestyle='--', alpha=0.6, label='Pet Catch Radius (0.8m)')
    ax.add_patch(pet_reach)

    # Styling
    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(0, ROOM_HEIGHT)
    ax.set_xlabel("Room X (Meters)")
    ax.set_ylabel("Room Y (Meters)")
    ax.set_title(title, fontsize=14, weight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_exploration_heatmap(visited_grid, title="Spatial Exploration Heatmap (20x20 Grid)", save_name="exploration_heatmap.png"):
    """Renders 20x20 spatial heatmap of patient room exploration."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    
    sns.heatmap(visited_grid, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Visit Count'},
                ax=ax, linewidths=0.5, linecolor='lightgray')

    # Draw Neglect Division Line (column 10)
    ax.axvline(10, color='blue', linestyle='--', linewidth=2, label='Neglect Boundary (x=5.0m)')
    
    ax.set_title(title, fontsize=13, weight='bold', pad=10)
    ax.set_xlabel("Grid Column (0 = Extreme Left, 19 = Extreme Right)")
    ax.set_ylabel("Grid Row (0 = Top, 19 = Bottom)")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_learning_curves(results_dict, save_name="learning_curves.png"):
    """
    Plots comparative learning curves for PPO, DQN, A2C algorithms across multiple metrics.
    
    Parameters:
        results_dict: dict mapping algo_name -> pandas DataFrame with 'episode', 'reward', 'total_expl', 'left_expl', 'success'
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    colors = {'PPO': '#1f77b4', 'DQN': '#ff7f0e', 'A2C': '#2ca02c'}

    for name, df in results_dict.items():
        c = colors.get(name, 'black')
        window = max(1, len(df) // 20)
        
        # 1. Average Reward
        smooth_r = df['reward'].rolling(window=window, min_periods=1).mean()
        axes[0, 0].plot(df['episode'], smooth_r, label=name, color=c, linewidth=2)
        axes[0, 0].set_title("Reward Progression", weight='bold')
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].grid(True)

        # 2. Total Exploration %
        smooth_tot = df['total_expl'].rolling(window=window, min_periods=1).mean() * 100.0
        axes[0, 1].plot(df['episode'], smooth_tot, label=name, color=c, linewidth=2)
        axes[0, 1].set_title("Total Room Exploration (%)", weight='bold')
        axes[0, 1].set_ylabel("Exploration %")
        axes[0, 1].grid(True)

        # 3. Left-Side Exploration %
        smooth_left = df['left_expl'].rolling(window=window, min_periods=1).mean() * 100.0
        axes[1, 0].plot(df['episode'], smooth_left, label=name, color=c, linewidth=2)
        axes[1, 0].set_title("Left (Neglected Side) Exploration (%)", weight='bold')
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Left Exploration %")
        axes[1, 0].grid(True)

        # 4. Success Rate %
        smooth_succ = df['success'].rolling(window=window, min_periods=1).mean() * 100.0
        axes[1, 1].plot(df['episode'], smooth_succ, label=name, color=c, linewidth=2)
        axes[1, 1].set_title("Rehabilitation Success Rate (%)", weight='bold')
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Success Rate %")
        axes[1, 1].grid(True)

    for ax in axes.flat:
        ax.legend(loc='lower right')

    plt.suptitle("Reinforcement Learning Training Comparison (PPO vs DQN vs A2C)", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_evaluation_comparison(summary_df, save_name="eval_benchmark_comparison.png"):
    """Creates summary bar plot comparing Baselines vs RL policies on key evaluation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    methods = summary_df['Method']
    
    # 1. Average Reward
    axes[0].bar(methods, summary_df['avg_reward'], color=['#7f7f7f', '#a65628', '#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title("Average Episode Reward", weight='bold')
    axes[0].set_ylabel("Reward")
    axes[0].tick_params(axis='x', rotation=30)

    # 2. Left Exploration %
    axes[1].bar(methods, summary_df['avg_left_expl_pct'], color=['#7f7f7f', '#a65628', '#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title("Left (Neglected Side) Exploration %", weight='bold')
    axes[1].set_ylabel("Left Coverage %")
    axes[1].tick_params(axis='x', rotation=30)

    # 3. Success Rate %
    axes[2].bar(methods, summary_df['success_rate_pct'], color=['#7f7f7f', '#a65628', '#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_title("Rehabilitation Success Rate %", weight='bold')
    axes[2].set_ylabel("Success %")
    axes[2].tick_params(axis='x', rotation=30)

    plt.suptitle("Evaluation Benchmark: Baseline Heuristics vs Trained RL Agents", fontsize=15, weight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def create_interactive_trajectory_plotly(patient_trajectory, pet_pos, obstacles, save_name="interactive_trajectory.html"):
    """Generates interactive Plotly HTML viewer for patient movement trajectory."""
    traj = np.array(patient_trajectory)
    fig = go.Figure()

    # Neglected side background shapes
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=10,
                  fillcolor="rgba(255, 0, 0, 0.15)", line=dict(width=0), name="Severe Left Neglect")
    fig.add_shape(type="rect", x0=2.5, y0=0, x1=5.0, y1=10,
                  fillcolor="rgba(255, 165, 0, 0.15)", line=dict(width=0), name="Moderate Left Neglect")

    # Obstacles
    for obs in obstacles:
        fig.add_shape(type="circle",
                      x0=obs['pos'][0]-obs['radius'], y0=obs['pos'][1]-obs['radius'],
                      x1=obs['pos'][0]+obs['radius'], y1=obs['pos'][1]+obs['radius'],
                      fillcolor="Brown", opacity=0.7, line=dict(color="Black"))

    # Patient Trajectory Path
    fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers',
                             line=dict(color='purple', width=3),
                             marker=dict(size=4), name='Patient Path'))

    # Start and End points
    fig.add_trace(go.Scatter(x=[traj[0, 0]], y=[traj[0, 1]], mode='markers',
                             marker=dict(size=12, color='blue'), name='Start Pos'))
    fig.add_trace(go.Scatter(x=[traj[-1, 0]], y=[traj[-1, 1]], mode='markers',
                             marker=dict(size=14, color='indigo'), name='End Pos'))

    # Digital Pet
    fig.add_trace(go.Scatter(x=[pet_pos[0]], y=[pet_pos[1]], mode='markers',
                             marker=dict(size=18, color='DeepPink', symbol='star'), name='Digital Pet 🐶'))

    fig.update_layout(
        title="AR Rehabilitation Interactive Patient Trajectory",
        xaxis=dict(title="Room X (m)", range=[0, 10]),
        yaxis=dict(title="Room Y (m)", range=[0, 10]),
        width=700, height=700
    )

    save_path = os.path.join(OUTPUT_DIR, save_name)
    fig.write_html(save_path)
    return save_path
