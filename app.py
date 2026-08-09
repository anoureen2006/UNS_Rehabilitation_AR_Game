"""
app.py
Streamlit Web Application for Augmented Reality (AR) Rehabilitation Digital Pet RL Simulation.
Allows interactive simulation, real-time visualization of stroke patient trajectory under USN neglect,
policy model selection (PPO, DQN, A2C, Baselines), learning analytics, and Unity AR export.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, OUTPUT_DIR, CHECKPOINT_DIR
from environment import ARRehabEnv
from patient import PatientSim
from digital_pet import DigitalPet
from reward import RehabilitationRewardCalculator
from utils import HeuristicPolicies, calculate_metrics, set_seeds
from visualization import (
    plot_room_and_trajectory,
    plot_exploration_heatmap,
    plot_evaluation_comparison,
    create_interactive_trajectory_plotly
)

from stable_baselines3 import PPO, DQN, A2C

# Page Configuration
st.set_page_config(
    page_title="AR Rehab RL - USN Digital Pet Simulation",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# Title Banner
st.markdown('<div class="main-header">AR Rehabilitation RL Simulation 🐶</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Spatial Target Placement for Stroke Patients with Unilateral Spatial Neglect (USN)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Patient & Environment Controls")

neglect_severity = st.sidebar.slider(
    "USN Neglect Severity (Left Side)",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="0.0 = Normal scanning; 1.0 = Severe left spatial neglect."
)

walking_speed = st.sidebar.slider(
    "Patient Base Speed (m/s)",
    min_value=0.3, max_value=1.5, value=0.8, step=0.1
)

fov_degrees = st.sidebar.slider(
    "Field of View Angle (°)",
    min_value=60, max_value=180, value=120, step=10
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 RL Agent & Policy Selection")

policy_choice = st.sidebar.selectbox(
    "Select Target Placement Policy:",
    ["PPO Agent (Trained)", "DQN Agent (Trained)", "A2C Agent (Trained)",
     "Baseline: Extreme Left Heuristic", "Baseline: Center Placement", "Baseline: Random Placement"]
)

seed_val = st.sidebar.number_input("Random Seed", value=42, step=1)

# Helper function to load trained model or return heuristic
@st.cache_resource
def load_rl_model(algo_name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{algo_name.lower()}_rehab_model.zip")
    if os.path.exists(ckpt_path):
        if algo_name == "PPO":
            return PPO.load(ckpt_path, device='cpu')
        elif algo_name == "DQN":
            return DQN.load(ckpt_path, device='cpu')
        elif algo_name == "A2C":
            return A2C.load(ckpt_path, device='cpu')
    return None

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🎮 Live Simulation", "📊 Interactive 3D & Trajectory",
    "📈 Algorithm Comparison", "📲 Unity AR Telemetry Export"
])

# ==============================================================================
# TAB 1: LIVE SIMULATION
# ==============================================================================
with tab1:
    col_ctrl, col_plot1, col_plot2 = st.columns([1, 2, 2])
    
    with col_ctrl:
        st.subheader("Episode Execution")
        run_ep_btn = st.button("🚀 Run Live Simulation Episode", type="primary", use_container_width=True)
        
    # Run episode on button click or default load
    if run_ep_btn or "last_env" not in st.session_state:
        set_seeds(seed_val)
        env = ARRehabEnv(neglect_severity=neglect_severity, seed=seed_val)
        # Update custom patient parameters
        env.patient.base_speed = walking_speed
        env.patient.fov_rad = np.radians(fov_degrees)
        
        obs, info = env.reset()
        
        # Determine action
        if "PPO" in policy_choice:
            model = load_rl_model("PPO")
            action = model.predict(obs, deterministic=True)[0] if model else HeuristicPolicies.extreme_left_policy()
        elif "DQN" in policy_choice:
            model = load_rl_model("DQN")
            action = model.predict(obs, deterministic=True)[0] if model else HeuristicPolicies.extreme_left_policy()
        elif "A2C" in policy_choice:
            model = load_rl_model("A2C")
            action = model.predict(obs, deterministic=True)[0] if model else HeuristicPolicies.extreme_left_policy()
        elif "Extreme Left" in policy_choice:
            action = HeuristicPolicies.extreme_left_policy()
        elif "Center" in policy_choice:
            action = HeuristicPolicies.center_policy()
        else:
            action = HeuristicPolicies.random_policy()

        # Step simulation until episode completion
        done = False
        ep_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        st.session_state["last_env"] = env
        st.session_state["last_info"] = step_info
        st.session_state["last_reward"] = ep_reward

    env = st.session_state["last_env"]
    step_info = st.session_state["last_info"]
    ep_reward = st.session_state["last_reward"]

    # Display Metrics Banner
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Episode Reward", f"{ep_reward:.2f}")
    m2.metric("Total Exploration", f"{step_info['total_expl_ratio']*100:.1f}%")
    m3.metric("Left (Neglected) Expl", f"{step_info['left_expl_ratio']*100:.1f}%")
    m4.metric("Time to Pet", f"{step_info['step_count']*0.5:.1f}s")
    
    pet_found = step_info["pet_found"]
    left_expl = step_info["left_expl_ratio"]
    status_str = "✅ Rehab Success" if (pet_found and left_expl >= 0.20) else ("⚠️ Premature Found" if pet_found else "❌ Timeout")
    m5.metric("Clinical Outcome", status_str)

    # Render Visualizations
    with col_plot1:
        st.subheader("2D Room Layout & Patient Path")
        fig_path = plot_room_and_trajectory(env, env.patient, env.pet,
                                            title=f"Policy: {policy_choice}",
                                            save_name="st_room_traj.png")
        st.image(fig_path, use_column_width=True)

    with col_plot2:
        st.subheader("20x20 Spatial Exploration Heatmap")
        heatmap_path = plot_exploration_heatmap(env.patient.visited_grid, save_name="st_heatmap.png")
        st.image(heatmap_path, use_column_width=True)

# ==============================================================================
# TAB 2: INTERACTIVE PLOTLY 3D & TRAJECTORY
# ==============================================================================
with tab2:
    st.subheader("Interactive Patient Trajectory & Spatial Exploration")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### Interactive Trajectory Viewer")
        html_path = create_interactive_trajectory_plotly(
            env.patient.trajectory, env.pet.pos, env.obstacles, save_name="st_interactive_traj.html"
        )
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)

    with col_p2:
        st.markdown("#### 3D Exploration Density Surface")
        grid_data = env.patient.visited_grid
        fig_3d = go.Figure(data=[go.Surface(z=grid_data, colorscale='YlOrRd')])
        fig_3d.update_layout(
            title="3D Room Exploration Density",
            scene=dict(
                xaxis_title="Room X (Cols 0-19)",
                yaxis_title="Room Y (Rows 0-19)",
                zaxis_title="Visit Frequency"
            ),
            width=600, height=550
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# TAB 3: ALGORITHM COMPARISON
# ==============================================================================
with tab3:
    st.subheader("Reinforcement Learning Algorithms Performance Benchmark")
    
    if st.button("📊 Run 50-Episode Comparative Benchmark"):
        with st.spinner("Running evaluation benchmark across 50 test episodes..."):
            from evaluate import run_benchmark
            eval_df = run_benchmark(num_episodes=50, seed=seed_val)
            st.session_state["eval_df"] = eval_df

    if "eval_df" in st.session_state:
        st.dataframe(st.session_state["eval_df"], use_container_width=True)
        eval_img = os.path.join(OUTPUT_DIR, "eval_benchmark_comparison.png")
        if os.path.exists(eval_img):
            st.image(eval_img, use_column_width=True)
    else:
        st.info("Click 'Run 50-Episode Comparative Benchmark' to generate comparison tables and charts.")

    st.markdown("---")
    st.subheader("Training Learning Curves (PPO vs DQN vs A2C)")
    curves_img = os.path.join(OUTPUT_DIR, "learning_curves_comparison.png")
    if os.path.exists(curves_img):
        st.image(curves_img, use_column_width=True)
    else:
        st.info("Run `python run_all.py` or training script to display full training curves.")

# ==============================================================================
# TAB 4: UNITY AR TELEMETRY EXPORT
# ==============================================================================
with tab4:
    st.subheader("Unity AR Rehabilitation Application Integration")
    st.markdown("""
    This RL backend generates digital pet coordinates and clinical telemetry ready for integration into a Unity AR headset application (e.g. Meta Quest / HoloLens / iOS ARKit).
    """)
    
    unity_data = {
        "sessionId": f"REHAB_USN_{seed_val}",
        "patientProperties": {
            "neglectSeverity": float(neglect_severity),
            "walkingSpeed": float(walking_speed),
            "fovDegrees": float(fov_degrees),
            "fatigue": float(env.patient.fatigue)
        },
        "digitalPetSpawnTarget": {
            "cellId": int(env.pet.cell_id),
            "unityPosition": {
                "x": float(env.pet.pos[0]),
                "y": 0.0, # floor height
                "z": float(env.pet.pos[1])
            }
        },
        "sessionMetrics": {
            "reward": float(ep_reward),
            "leftExplorationPercentage": float(step_info['left_expl_ratio'] * 100.0),
            "totalExplorationPercentage": float(step_info['total_expl_ratio'] * 100.0),
            "timeToPetSeconds": float(step_info['step_count'] * 0.5),
            "rehabilitationSuccess": bool(step_info['pet_found'] and step_info['left_expl_ratio'] >= 0.20)
        }
    }
    
    st.json(unity_data)
    st.download_button(
        label="📥 Download Unity AR Telemetry JSON",
        data=pd.Series(unity_data).to_json(indent=2),
        file_name="unity_ar_rehab_target.json",
        mime="application/json"
    )
