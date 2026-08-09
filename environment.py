"""
environment.py
Gymnasium Environment for AR Augmented Reality Stroke Rehabilitation digital pet placement.
Models procedural room layout, patient exploration physics, USN neglect dynamics,
and state/action spaces for Stable-Baselines3 agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, NUM_CELLS,
    CELL_WIDTH, CELL_HEIGHT, MIN_OBSTACLES, MAX_OBSTACLES,
    OBSTACLE_TYPES, MAX_EPISODE_STEPS, DELTA_T, LEFT_BOUND, RANDOM_SEED
)
from patient import PatientSim
from digital_pet import DigitalPet
from reward import RehabilitationRewardCalculator

class ARRehabEnv(gym.Env):
    """
    Gymnasium AR Rehabilitation Environment.
    The RL agent chooses WHERE to spawn the digital pet (Action: cell ID 0..399 or continuous (x,y)).
    The environment simulates the patient's exploratory trajectory step-by-step
    under Unilateral Spatial Neglect (USN) constraints.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, neglect_severity=0.7, use_continuous_action=False, seed=None):
        super(ARRehabEnv, self).__init__()

        self.seed_val = seed if seed is not None else RANDOM_SEED
        self.rng = np.random.default_rng(self.seed_val)
        self.use_continuous_action = use_continuous_action
        
        # Action Space
        if self.use_continuous_action:
            self.action_space = spaces.Box(
                low=np.array([0.5, 0.5], dtype=np.float32),
                high=np.array([9.5, 9.5], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(NUM_CELLS)  # 400 cells (0 to 399)

        # Observation Space Vector Dimension = 44 features
        self.obs_dim = 44
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Core Components
        self.patient = PatientSim(neglect_severity=neglect_severity, seed=self.seed_val)
        self.pet = DigitalPet()
        self.reward_calculator = RehabilitationRewardCalculator()
        
        # Room & Episode State
        self.obstacles = []
        self.step_count = 0
        self.episode_num = 0
        self.current_difficulty = 1.0
        self.prev_pet_pos = np.array([5.0, 5.0], dtype=np.float32)
        self.prev_reward = 0.0
        self.episode_rewards = []
        self.prev_visited_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)

    def _generate_random_room(self):
        """Procedurally places chairs, tables, books, cups, plants, doors, windows in 10x10 room."""
        self.obstacles = []
        num_obs = self.rng.integers(MIN_OBSTACLES, MAX_OBSTACLES + 1)
        
        for _ in range(num_obs):
            obs_type = self.rng.choice(OBSTACLE_TYPES)
            # Size and radius depending on object type
            if obs_type in ['chair', 'plant']:
                radius = 0.45
            elif obs_type == 'table':
                radius = 0.75
            elif obs_type in ['door', 'window']:
                radius = 0.2  # near perimeter walls
            else:
                radius = 0.25 # book, cup
                
            # Random position avoiding extreme initial patient start area (5.0, 1.5)
            while True:
                pos = self.rng.uniform([0.8, 0.8], [ROOM_WIDTH - 0.8, ROOM_HEIGHT - 0.8])
                if np.linalg.norm(pos - np.array([5.0, 1.5])) > 1.2:
                    break
                    
            self.obstacles.append({
                "type": obs_type,
                "pos": pos,
                "radius": radius
            })

    def reset(self, seed=None, options=None):
        """Resets room layout, patient position, and episode statistics."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_num += 1
        self.step_count = 0
        
        # Procedurally generate new room layout
        self._generate_random_room()
        
        # Vary patient initial position and neglect severity slightly for domain randomization
        init_x = self.rng.uniform(4.5, 8.0) # Patient starts on right/center side
        init_y = self.rng.uniform(1.0, 2.5)
        neglect_sev = self.rng.uniform(0.5, 0.9)
        
        self.patient.reset(init_pos=[init_x, init_y], neglect_severity=neglect_sev)
        self.prev_visited_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        
        # Set default pet position before action (or center)
        self.pet.set_position(self.prev_pet_pos)
        
        obs = self._get_observation()
        info = {
            "episode": self.episode_num,
            "neglect_severity": neglect_sev,
            "num_obstacles": len(self.obstacles)
        }
        return obs, info

    def set_pet_action(self, action):
        """Places the digital pet in the room according to the RL action."""
        if self.use_continuous_action:
            pet_target = action
        else:
            pet_target = int(action)
            
        self.pet.set_position(pet_target)
        # Ensure pet isn't trapped inside an obstacle
        if not DigitalPet.is_valid_location(self.pet.pos, self.obstacles):
            # Shift pet slightly to nearest open space
            self.pet.pos[0] = np.clip(self.pet.pos[0] + 0.4, 0.5, 9.5)
            self.pet.pos[1] = np.clip(self.pet.pos[1] + 0.4, 0.5, 9.5)

    def step(self, action):
        """
        Executes one step in the environment.
        1. Pet location is updated based on action (if first step or step-by-step guidance).
        2. Patient moves according to cognitive USN motion model.
        3. Rewards and episode termination conditions are calculated.
        """
        # On step 0, place the digital pet specified by RL agent
        if self.step_count == 0:
            self.set_pet_action(action)

        self.step_count += 1
        prev_pos = self.patient.pos.copy()
        
        # Advance patient simulation 1 step
        patient_res = self.patient.step(self.pet.pos, self.obstacles)
        patient_res["prev_pos"] = prev_pos
        patient_res["visited_grid"] = self.patient.visited_grid.copy()
        patient_res["neglect_severity"] = self.patient.neglect_severity

        # Calculate reward
        reward, reward_info = self.reward_calculator.compute_step_reward(
            patient_state=patient_res,
            pet_pos=self.pet.pos,
            prev_visited_grid=self.prev_visited_grid,
            step_count=self.step_count,
            max_steps=MAX_EPISODE_STEPS
        )

        # Update previous visited snapshot
        self.prev_visited_grid = self.patient.visited_grid.copy()
        self.prev_reward = reward

        # Check termination & truncation
        pet_found = patient_res["pet_found"]
        terminated = pet_found
        truncated = (self.step_count >= MAX_EPISODE_STEPS)

        obs = self._get_observation()
        
        info = {
            **reward_info,
            "pet_found": pet_found,
            "pet_pos": self.pet.pos.copy(),
            "patient_pos": self.patient.pos.copy(),
            "fatigue": self.patient.fatigue,
            "neglect_severity": self.patient.neglect_severity
        }

        if terminated or truncated:
            self.prev_pet_pos = self.pet.pos.copy()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Constructs a 44-dimensional normalized vector representing the current state."""
        # 1. Patient Position (2) -> normalized [0, 1]
        norm_pos = self.patient.pos / np.array([ROOM_WIDTH, ROOM_HEIGHT])
        
        # 2. Heading Angle (2) -> sin/cos orientation
        heading_vec = np.array([np.cos(self.patient.heading), np.sin(self.patient.heading)])
        
        # 3. Exploration Matrix Downsampled (5x5 grid = 25)
        grid = self.patient.visited_grid.copy()
        # Pool 20x20 into 5x5 by taking 4x4 block means
        grid_5x5 = grid.reshape(5, 4, 5, 4).mean(axis=(1, 3))
        norm_grid_5x5 = np.clip(grid_5x5.flatten() / 3.0, 0.0, 1.0)
        
        # 4. Global Exploration Metrics (4)
        visited_cells = (grid > 0).astype(np.float32)
        total_expl = np.mean(visited_cells)
        left_expl = np.mean(visited_cells[:, :GRID_COLS//2])
        right_expl = np.mean(visited_cells[:, GRID_COLS//2:])
        asymmetry = abs(left_expl - right_expl) / (left_expl + right_expl + 1e-6)
        expl_metrics = np.array([total_expl, left_expl, right_expl, asymmetry], dtype=np.float32)
        
        # 5. Patient Internal State (4)
        internal_state = np.array([
            self.patient.neglect_severity,
            self.patient.fatigue,
            self.patient.attention,
            self.patient.base_speed / 1.5
        ], dtype=np.float32)
        
        # 6. Time & Episode Info (4)
        t_ratio = self.step_count / float(MAX_EPISODE_STEPS)
        rem_time_ratio = 1.0 - t_ratio
        diff_level = self.current_difficulty / 5.0
        ep_ratio = min(1.0, self.episode_num / 1000.0)
        time_info = np.array([t_ratio, rem_time_ratio, diff_level, ep_ratio], dtype=np.float32)
        
        # 7. Previous Pet Position & Prev Reward (3)
        norm_prev_pet = self.prev_pet_pos / np.array([ROOM_WIDTH, ROOM_HEIGHT])
        scaled_prev_r = np.clip(self.prev_reward / 30.0, -1.0, 1.0)
        prev_info = np.array([norm_prev_pet[0], norm_prev_pet[1], scaled_prev_r], dtype=np.float32)

        # Concatenate into 44-dim normalized state vector
        obs = np.concatenate([
            norm_pos,          # 2
            heading_vec,       # 2
            norm_grid_5x5,     # 25
            expl_metrics,      # 4
            internal_state,    # 4
            time_info,         # 4
            prev_info          # 3
        ]).astype(np.float32)

        return obs
