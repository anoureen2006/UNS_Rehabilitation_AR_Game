"""
app.py
Streamlit Web Application for Augmented Reality (AR) Rehabilitation Digital Pet RL Simulation.
Supports live 2D/3D environment exploration, model benchmarking (PPO, DQN, A2C),
and direct Unity Telemetry Session JSON uploading & RL trial adaptation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import json
import os

from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, OUTPUT_DIR, CHECKPOINT_DIR
from environment import ARRehabEnv
from patient import PatientSim
from digital_pet import DigitalPet
from reward import RehabilitationRewardCalculator
from utils import HeuristicPolicies, calculate_metrics, set_seeds
from visualization import (
    plot_room_and_trajectory,
    plot_exploration_heatmap,
    create_interactive_trajectory_plotly
)

from unity_schema import UnitySession
from unity_dataset import session_to_observation, predict_next_difficulty
from unity_env import UnityARRehabEnv

from stable_baselines3 import PPO, DQN, A2C

# Page Configuration
st.set_page_config(
    page_title="AR Rehab RL - USN Unity Adaptive Controller",
    page_icon="🎮",
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
st.markdown('<div class="main-header">Unity AR Rehabilitation RL Controller 🎮🐶</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Adaptive Difficulty & Target Placement for Unilateral Spatial Neglect (USN) Stroke Patients</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Patient & Simulation Controls")

neglect_severity = st.sidebar.slider(
    "USN Neglect Severity (Left Side)",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05
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
st.sidebar.header("🤖 Policy Selection")

policy_choice = st.sidebar.selectbox(
    "Select Target Placement Policy:",
    ["PPO Agent (Trained)", "DQN Agent (Trained)", "A2C Agent (Trained)",
     "Baseline: Extreme Left Heuristic", "Baseline: Center Placement", "Baseline: Random Placement"]
)

seed_val = st.sidebar.number_input("Random Seed", value=42, step=1)

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
tab_unity, tab_sim, tab_interactive, tab_bench = st.tabs([
    "📥 Unity JSON Session Telemetry", "🎮 2D Live Room Simulation",
    "📊 3D Trajectory Analytics", "📈 Algorithm Comparison"
])

# ==============================================================================
# TAB 1: UNITY JSON SESSION TELEMETRY (DIRECT USER DATA INPUT)
# ==============================================================================
with tab_unity:
    st.subheader("📥 Upload & Analyze Unity Game Session Log")
    st.markdown("""
    Upload your Unity game session JSON file (or paste its contents) containing `session_id`, `patient_id`, `neglect_side`, and `trials` list.
    The RL controller will analyze performance and compute the **optimal difficulty parameters for the next trial**.
    """)

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        uploaded_file = st.file_uploader("Upload Unity Session `.json` File", type=["json"])
        
    with col_input2:
        default_json = ""
        sample_path = "sample_unity_session.json"
        if os.path.exists(sample_path):
            with open(sample_path, 'r', encoding='utf-8') as f:
                default_json = f.read()
        pasted_json = st.text_area("Or Paste Raw Unity Session JSON Here:", value=default_json, height=180)

    # Process JSON data
    raw_json_str = None
    if uploaded_file is not None:
        raw_json_str = uploaded_file.getvalue().decode("utf-8")
    elif pasted_json.strip():
        raw_json_str = pasted_json.strip()

    if raw_json_str:
        try:
            session = UnitySession.from_json_str(raw_json_str)
            st.success(f"Successfully loaded Unity Session `{session.session_id}` for patient `{session.patient_id}` ({len(session.trials)} trials recorded).")

            # Compute RL prediction
            prediction = predict_next_difficulty(raw_json_str)
            next_diff = prediction["recommended_next_trial"]["difficulty_at_trial"]

            # Key Summary Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            hits = [t.hit for t in session.trials]
            rts = [t.reaction_time_ms for t in session.trials if t.hit]
            neg_hits = [t for t in session.trials if t.hemifield == "neglected"]
            
            c1.metric("Total Trials", len(session.trials))
            c2.metric("Hit Rate", f"{(sum(hits)/len(hits)*100):.1f}%" if hits else "0%")
            c3.metric("Avg Hit Reaction Time", f"{np.mean(rts):.0f} ms" if rts else "N/A")
            c4.metric("Neglected Hemifield Hits", f"{sum(1 for t in neg_hits if t.hit)} / {len(neg_hits)}")
            c5.metric("Timeouts (30s)", sum(1 for t in session.trials if t.reaction_time_ms >= 30000.0))

            st.markdown("---")
            st.subheader("🤖 RL Policy Recommendation for Next Trial")
            
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Recommended Speed", f"{next_diff['speed']:.2f} m/s", delta="Adaptive")
            p2.metric("Recommended Eccentricity", f"{next_diff['eccentricity_deg']:.1f}°", delta="Scaffolded")
            p3.metric("Recommended Distance", f"{next_diff['distance_m']:.2f} m")
            p4.metric("Target Count & Limit", f"{next_diff['target_count']} targets ({next_diff['time_limit_s']:.0f}s)")

            # Trial Performance Charts
            st.markdown("#### Unity Trial Performance Telemetry")
            df_trials = pd.DataFrame([
                {
                    "Trial #": i + 1,
                    "Hit": t.hit,
                    "Reaction Time (s)": t.reaction_time_ms / 1000.0,
                    "Gaze Offset (°)": t.gaze_angle_deg,
                    "Hemifield": t.hemifield,
                    "Eccentricity (°)": t.difficulty_at_trial.eccentricity_deg,
                    "Speed (m/s)": t.difficulty_at_trial.speed
                }
                for i, t in enumerate(session.trials)
            ])

            fig_rt = px.line(df_trials, x="Trial #", y="Reaction Time (s)", color="Hit",
                             symbol="Hemifield", title="Reaction Time & Trial Outcome Across Session")
            st.plotly_chart(fig_rt, use_container_width=True)

            # Download Recommendation JSON
            st.download_button(
                label="📥 Download RL Recommended Next Trial Settings (JSON)",
                data=json.dumps(prediction, indent=2),
                file_name=f"rl_recommendation_{session.session_id}.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error parsing Unity JSON format: {e}")

# ==============================================================================
# TAB 2: LIVE 2D ROOM SIMULATION
# ==============================================================================
with tab_sim:
    col_ctrl, col_plot1, col_plot2 = st.columns([1, 2, 2])
    with col_ctrl:
        st.subheader("Episode Controls")
        run_ep_btn = st.button("🚀 Run Simulation Episode", type="primary", use_container_width=True)

    if run_ep_btn or "last_env" not in st.session_state:
        set_seeds(seed_val)
        env = ARRehabEnv(neglect_severity=neglect_severity, seed=seed_val)
        env.patient.base_speed = walking_speed
        env.patient.fov_rad = np.radians(fov_degrees)
        obs, info = env.reset()

        if "PPO" in policy_choice:
            model = load_rl_model("PPO")
            action = model.predict(obs, deterministic=True)[0] if model else HeuristicPolicies.extreme_left_policy()
        else:
            action = HeuristicPolicies.extreme_left_policy()

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

    with col_plot1:
        st.subheader("2D Room Layout & Path")
        fig_path = plot_room_and_trajectory(env, env.patient, env.pet,
                                            title=f"Policy: {policy_choice}", save_name="st_room_traj.png")
        st.image(fig_path, use_column_width=True)

    with col_plot2:
        st.subheader("20x20 Spatial Exploration Heatmap")
        heatmap_path = plot_exploration_heatmap(env.patient.visited_grid, save_name="st_heatmap.png")
        st.image(heatmap_path, use_column_width=True)

# ==============================================================================
# TAB 3: 3D TRAJECTORY ANALYTICS
# ==============================================================================
with tab_interactive:
    st.subheader("Interactive 3D Spatial Trajectory")
    grid_data = env.patient.visited_grid
    fig_3d = go.Figure(data=[go.Surface(z=grid_data, colorscale='YlOrRd')])
    fig_3d.update_layout(title="3D Room Exploration Density", width=800, height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# TAB 4: ALGORITHM COMPARISON
# ==============================================================================
with tab_bench:
    st.subheader("Algorithm Comparison Dashboard")
    curves_img = os.path.join(OUTPUT_DIR, "learning_curves_comparison.png")
    if os.path.exists(curves_img):
        st.image(curves_img, use_column_width=True)
