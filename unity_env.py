"""
unity_env.py
Gymnasium Environment designed specifically around Unity AR Rehabilitation telemetry format.
Inputs: Exact trial history JSON structure (hit, reaction_time_ms, gaze_angle_deg, hemifield, difficulty).
Outputs: Optimal difficulty_at_trial for 4 RL actions: (speed, eccentricity_deg, distance_m, time_limit_s).
Target count remains fixed at 3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional

from unity_schema import UnitySession, UnityTrial, DifficultyAtTrial

# Predefined discrete difficulty grids for Unity AR game
SPEED_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6]                   # m/s (5 choices)
ECCENTRICITY_LEVELS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]    # degrees (6 choices)
DISTANCE_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0]               # meters (5 choices)
TIME_LIMIT_LEVELS = [10.0, 15.0, 20.0, 30.0, 45.0]        # seconds (5 choices)
# Total Discrete Actions = 5 * 6 * 5 * 5 = 750

class UnityARRehabEnv(gym.Env):
    """
    Gymnasium Environment using exact Unity AR Rehabilitation Telemetry schema.
    Observation Vector: 15 normalized state features matching session log trials.
    Action Space: Selects next trial's difficulty (speed, eccentricity, distance, time_limit_s).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, use_continuous_action=False, seed=42):
        super(UnityARRehabEnv, self).__init__()
        self.rng = np.random.default_rng(seed)
        self.use_continuous_action = use_continuous_action

        # Action Space (4 Actions: speed, eccentricity_deg, distance_m, time_limit_s)
        if self.use_continuous_action:
            self.action_space = spaces.Box(
                low=np.array([0.2, 5.0, 0.8, 10.0], dtype=np.float32),
                high=np.array([0.8, 35.0, 2.5, 45.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.num_actions = len(SPEED_LEVELS) * len(ECCENTRICITY_LEVELS) * len(DISTANCE_LEVELS) * len(TIME_LIMIT_LEVELS)
            self.action_space = spaces.Discrete(self.num_actions)

        # Observation Space (15 normalized state features)
        self.obs_dim = 15
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Internal Trial & Patient Simulation State
        self.current_trial_idx = 0
        self.max_trials = 60
        self.neglect_severity = 0.7
        self.fatigue = 0.0
        self.consecutive_timeouts = 0
        self.trial_history = []
        
        self.current_diff = DifficultyAtTrial(
            speed=0.3, eccentricity_deg=10.0, distance_m=1.0, target_count=3, time_limit_s=30.0
        )

    def action_to_difficulty(self, action) -> DifficultyAtTrial:
        """Decodes integer or continuous action into Unity DifficultyAtTrial object."""
        if self.use_continuous_action:
            # Properly scale continuous actions from [-1.0, 1.0] to parameter ranges
            act0 = float(np.clip(action[0], -1.0, 1.0))
            act1 = float(np.clip(action[1], -1.0, 1.0))
            act2 = float(np.clip(action[2], -1.0, 1.0))
            act3 = float(np.clip(action[3], -1.0, 1.0))

            speed = float(0.2 + 0.5 * (act0 + 1.0) * (0.8 - 0.2))
            ecc = float(5.0 + 0.5 * (act1 + 1.0) * (35.0 - 5.0))
            dist = float(0.8 + 0.5 * (act2 + 1.0) * (2.5 - 0.8))
            tlimit = float(10.0 + 0.5 * (act3 + 1.0) * (45.0 - 10.0))
        else:
            act_idx = int(action) % self.num_actions
            
            s_idx = act_idx // (len(ECCENTRICITY_LEVELS) * len(DISTANCE_LEVELS) * len(TIME_LIMIT_LEVELS))
            rem1 = act_idx % (len(ECCENTRICITY_LEVELS) * len(DISTANCE_LEVELS) * len(TIME_LIMIT_LEVELS))
            e_idx = rem1 // (len(DISTANCE_LEVELS) * len(TIME_LIMIT_LEVELS))
            rem2 = rem1 % (len(DISTANCE_LEVELS) * len(TIME_LIMIT_LEVELS))
            d_idx = rem2 // len(TIME_LIMIT_LEVELS)
            t_idx = rem2 % len(TIME_LIMIT_LEVELS)

            speed = SPEED_LEVELS[s_idx]
            ecc = ECCENTRICITY_LEVELS[e_idx]
            dist = DISTANCE_LEVELS[d_idx]
            tlimit = TIME_LIMIT_LEVELS[t_idx]

        return DifficultyAtTrial(
            speed=speed,
            eccentricity_deg=ecc,
            distance_m=dist,
            target_count=3,           # Fixed at 3
            time_limit_s=tlimit       # Dynamic RL Action
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new patient session."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_trial_idx = 0
        self.fatigue = 0.0
        self.consecutive_timeouts = 0
        self.neglect_severity = float(self.rng.uniform(0.5, 0.9))
        self.trial_history = []
        self.current_diff = DifficultyAtTrial(speed=0.3, eccentricity_deg=10.0, distance_m=1.0, target_count=3, time_limit_s=30.0)

        obs = self._get_observation()
        info = {
            "session_id": "sim_session",
            "neglect_severity": self.neglect_severity
        }
        return obs, info

    def simulate_patient_trial(self, diff: DifficultyAtTrial) -> UnityTrial:
        """Simulates trial execution based on difficulty (speed, eccentricity, distance, time_limit_s)."""
        is_neglected = self.rng.random() < 0.75
        hemifield = "neglected" if is_neglected else "non_neglected"

        ecc = diff.eccentricity_deg
        spd = diff.speed
        dist = diff.distance_m
        tlimit = diff.time_limit_s
        
        if is_neglected:
            # Realistic clinical perceptual probability equation
            ecc_penalty = 0.45 * (ecc / 30.0)
            speed_penalty = 0.20 * (spd / 0.5)
            fatigue_penalty = 0.15 * self.fatigue
            percept_prob = 1.0 - (self.neglect_severity * ecc_penalty) - speed_penalty - fatigue_penalty
            percept_prob = float(np.clip(percept_prob, 0.20, 0.95))
        else:
            percept_prob = float(np.clip(0.98 - 0.15 * self.fatigue, 0.50, 0.98))

        hit = bool(self.rng.random() < percept_prob)

        if hit:
            base_rt = 350.0 + 80.0 * (ecc / 10.0) + 100.0 * spd + 400.0 * self.fatigue
            rt_ms = float(np.clip(self.rng.normal(base_rt, 120.0), 250.0, tlimit * 1000.0 - 50.0))
            gaze_angle = float(self.rng.normal(ecc * 0.6, 2.0))
            self.consecutive_timeouts = 0
        else:
            # Timeout at trial time limit
            rt_ms = tlimit * 1000.0
            gaze_angle = float(self.rng.normal(ecc * 1.5, 5.0))
            self.consecutive_timeouts += 1

        self.fatigue = float(np.clip(self.fatigue + 0.012, 0.0, 1.0))

        return UnityTrial(
            timestamp=float(self.current_trial_idx * 2.5),
            hit=hit,
            reaction_time_ms=rt_ms,
            gaze_angle_deg=gaze_angle,
            hemifield=hemifield,
            difficulty_at_trial=diff
        )

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one trial step in the environment."""
        self.current_trial_idx += 1
        self.current_diff = self.action_to_difficulty(action)
        
        trial = self.simulate_patient_trial(self.current_diff)
        self.trial_history.append(trial)

        reward, r_info = self.compute_unity_reward(trial)

        terminated = (self.current_trial_idx >= self.max_trials)
        truncated = False

        obs = self._get_observation()
        
        info = {
            **r_info,
            "trial": trial,
            "fatigue": self.fatigue,
            "consecutive_timeouts": self.consecutive_timeouts,
            "trial_idx": self.current_trial_idx
        }

        return obs, reward, terminated, truncated, info

    def compute_unity_reward(self, trial: UnityTrial) -> Tuple[float, Dict[str, Any]]:
        """Computes reward for trial parameters including dynamic time limit."""
        ecc = trial.difficulty_at_trial.eccentricity_deg
        tlimit_ms = trial.difficulty_at_trial.time_limit_s * 1000.0
        is_neglected = (trial.hemifield == "neglected")
        
        r_hit = 0.0
        p_timeout = 0.0
        p_rt = 0.0

        if trial.hit:
            if is_neglected:
                ecc_multiplier = 1.0 + (ecc / 15.0)
                r_hit = 15.0 * ecc_multiplier
            else:
                r_hit = 3.0
                
            if trial.reaction_time_ms > 4000.0:
                p_rt = - (trial.reaction_time_ms - 4000.0) / 1000.0
        else:
            if trial.reaction_time_ms >= tlimit_ms:
                p_timeout = -15.0
            else:
                p_timeout = -5.0

        total_reward = r_hit + p_timeout + p_rt
        return total_reward, {
            "r_hit": float(r_hit),
            "p_timeout": float(p_timeout),
            "p_rt": float(p_rt),
            "total_reward": float(total_reward)
        }

    def _get_observation(self) -> np.ndarray:
        """Constructs 15-dimensional normalized vector matching Unity session data."""
        if not self.trial_history:
            return np.array([
                0.0, 0.0, 0.0, 0.0,
                0.3, 10.0/45.0, 1.0/3.0, 3.0/5.0, 30.0/45.0,
                0.5, 0.2, 0.0, 1.0, 0.0, 0.0
            ], dtype=np.float32)

        last_trial = self.trial_history[-1]
        diff = last_trial.difficulty_at_trial

        hit_val = 1.0 if last_trial.hit else 0.0
        rt_norm = np.clip(last_trial.reaction_time_ms / 30000.0, 0.0, 1.0)
        gaze_norm = np.clip(last_trial.gaze_angle_deg / 90.0, 0.0, 1.0)
        hemifield_neg = 1.0 if last_trial.hemifield == "neglected" else 0.0

        spd_norm = np.clip(diff.speed / 1.0, 0.0, 1.0)
        ecc_norm = np.clip(diff.eccentricity_deg / 45.0, 0.0, 1.0)
        dist_norm = np.clip(diff.distance_m / 3.0, 0.0, 1.0)
        cnt_norm = np.clip(diff.target_count / 5.0, 0.0, 1.0)
        tlimit_norm = np.clip(diff.time_limit_s / 45.0, 0.0, 1.0)

        recent_trials = self.trial_history[-5:]
        rolling_hit = np.mean([1.0 if t.hit else 0.0 for t in recent_trials])
        rolling_rt = np.mean([t.reaction_time_ms for t in recent_trials]) / 30000.0
        timeout_cnt = np.clip(self.consecutive_timeouts / 5.0, 0.0, 1.0)

        neglect_side_left = 1.0
        progress = np.clip(self.current_trial_idx / float(self.max_trials), 0.0, 1.0)
        fatigue = np.clip(self.fatigue, 0.0, 1.0)

        return np.array([
            hit_val, rt_norm, gaze_norm, hemifield_neg,
            spd_norm, ecc_norm, dist_norm, cnt_norm, tlimit_norm,
            rolling_hit, rolling_rt, timeout_cnt,
            neglect_side_left, progress, fatigue
        ], dtype=np.float32)
