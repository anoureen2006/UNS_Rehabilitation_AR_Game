"""
config.py
Configuration file for the AR Rehabilitation Digital Pet RL Simulation.
Contains room layout specs, patient cognitive/physical parameters, RL observation/action specs,
reward shaping hyperparams, and training settings.
"""

import os

# ==============================================================================
# ROOM & GRID SPECIFICATIONS
# ==============================================================================
ROOM_WIDTH = 10.0      # meters (X dimension)
ROOM_HEIGHT = 10.0     # meters (Y dimension)
GRID_COLS = 20         # 20 cells along X -> cell width = 0.5m
GRID_ROWS = 20         # 20 cells along Y -> cell height = 0.5m
NUM_CELLS = GRID_COLS * GRID_ROWS  # 400 total cells (Cell IDs: 0 to 399)

CELL_WIDTH = ROOM_WIDTH / GRID_COLS    # 0.5m
CELL_HEIGHT = ROOM_HEIGHT / GRID_ROWS  # 0.5m

# Room Object Types & Spawn Densities
OBSTACLE_TYPES = ['chair', 'table', 'book', 'cup', 'plant', 'door', 'window']
MIN_OBSTACLES = 6
MAX_OBSTACLES = 12

# Left/Right Spatial Boundary (x < 5.0 is Neglected/Left Side)
LEFT_BOUND = 5.0
EXTREME_LEFT_BOUND = 2.5

# ==============================================================================
# PATIENT SIMULATION PARAMETERS
# ==============================================================================
PATIENT_DEFAULTS = {
    "speed": 0.8,                # base walking speed (m/s)
    "fov_deg": 120.0,            # field of view cone angle in degrees
    "max_view_dist": 2.5,        # max visual perception distance in meters
    "reach_radius": 0.8,         # distance to catch/reach pet (meters)
    "neglect_severity": 0.7,     # USN severity: 0.0 (normal) to 1.0 (severe left neglect)
    "fatigue": 0.0,              # initial fatigue (0.0 to 1.0)
    "fatigue_rate": 0.0015,      # fatigue build-up per step
    "attention": 0.85,           # base attention / focus level
    "memory_decay": 0.95,        # spatial memory retention factor
    "reaction_time": 0.3,        # reaction delay in seconds
    "right_bias_weight": 0.6,    # directional rotation bias strength due to USN
}

# Time Step Settings
DELTA_T = 0.5                    # simulation step resolution (seconds)
MAX_EPISODE_STEPS = 200          # 200 steps * 0.5s = 100 seconds max per episode

# ==============================================================================
# REWARD SHAPING PARAMETERS
# ==============================================================================
REWARD_CONFIG = {
    "new_cell_visited": 0.5,             # bonus for discovering unvisited cell
    "left_side_multiplier": 1.5,         # extra multiplier for left-side cells
    "extreme_left_bonus": 2.0,           # bonus for exploring deep left (x < 2.5)
    "pet_found_base": 20.0,              # base reward when pet is found after useful exploration
    "rehab_success_left_scale": 15.0,    # extra bonus scaled by left exploration ratio
    "premature_found_penalty": -15.0,    # penalty if pet found before patient explores (>10 steps or <15% left expl)
    "right_bias_penalty": -0.3,          # penalty for steps moving strictly rightward without left turn
    "standing_still_penalty": -0.4,      # penalty for zero movement
    "repeated_visit_penalty": -0.1,      # small penalty for revisiting recent cells
    "timeout_penalty": -10.0,            # penalty if max time expires without pet found
    "step_penalty": -0.05,               # small step time penalty to encourage goal-oriented movement
}

# ==============================================================================
# RL TRAINING PARAMETERS
# ==============================================================================
RANDOM_SEED = 42
TOTAL_TIMESTEPS = 250000          # total steps for SB3 algorithms (~1250 episodes)
CHECKPOINT_FREQ = 25000           # save checkpoint every N steps
EVAL_EPISODES = 50                # evaluation episodes per policy

# Output Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
