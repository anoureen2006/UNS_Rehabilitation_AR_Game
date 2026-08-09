"""
patient.py
Simulates a stroke patient with Unilateral Spatial Neglect (USN).
Includes mathematical models for physical motion, visual field of view (FOV),
USN perceptual attenuation, fatigue build-up, working memory, and obstacle avoidance.
"""

import numpy as np
import math
from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, DELTA_T, PATIENT_DEFAULTS

class PatientSim:
    """
    Simulates a stroke patient exploring a 2D room.
    The patient suffers from Unilateral Spatial Neglect (USN), creating a strong
    perceptual and motor bias towards the right side of their visual field, while
    frequently ignoring stimuli or frontiers on their left side.
    """
    def __init__(self, init_pos=None, neglect_severity=0.7, seed=None):
        self.rng = np.random.default_rng(seed)
        
        # Initial Position (default: near bottom center / right)
        if init_pos is None:
            self.pos = np.array([5.0, 1.5], dtype=np.float32)
        else:
            self.pos = np.array(init_pos, dtype=np.float32)
            
        # Physical Properties
        self.base_speed = PATIENT_DEFAULTS["speed"]
        self.heading = self.rng.uniform(0, 2 * np.pi)  # direction angle in radians
        self.fov_rad = np.radians(PATIENT_DEFAULTS["fov_deg"])
        self.max_view_dist = PATIENT_DEFAULTS["max_view_dist"]
        self.reach_radius = PATIENT_DEFAULTS["reach_radius"]
        
        # Cognitive & USN Properties
        self.neglect_severity = np.clip(neglect_severity, 0.0, 1.0)
        self.fatigue = PATIENT_DEFAULTS["fatigue"]
        self.fatigue_rate = PATIENT_DEFAULTS["fatigue_rate"]
        self.attention = PATIENT_DEFAULTS["attention"]
        self.right_bias_weight = PATIENT_DEFAULTS["right_bias_weight"]
        
        # Trajectory & Memory Tracking
        self.trajectory = [self.pos.copy()]
        self.visited_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        self._update_visited_cell()

    def reset(self, init_pos=None, neglect_severity=None):
        """Reset patient state for a new episode."""
        if init_pos is None:
            self.pos = np.array([5.0, 1.5], dtype=np.float32)
        else:
            self.pos = np.array(init_pos, dtype=np.float32)
            
        if neglect_severity is not None:
            self.neglect_severity = np.clip(neglect_severity, 0.0, 1.0)
            
        self.heading = self.rng.uniform(-np.pi, np.pi)
        self.fatigue = 0.0
        self.trajectory = [self.pos.copy()]
        self.visited_grid.fill(0)
        self._update_visited_cell()

    def _update_visited_cell(self):
        """Mark current position in the 20x20 grid map."""
        col = int(np.clip(self.pos[0] / (ROOM_WIDTH / GRID_COLS), 0, GRID_COLS - 1))
        row = int(np.clip(self.pos[1] / (ROOM_HEIGHT / GRID_ROWS), 0, GRID_ROWS - 1))
        self.visited_grid[row, col] += 1

    def can_perceive(self, target_pos):
        """
        Calculates if the patient perceives a target position based on distance,
        FOV cone, and USN perceptual attenuation.
        """
        vec = target_pos - self.pos
        dist = np.linalg.norm(vec)
        
        # 1. Distance check
        if dist > self.max_view_dist:
            return False
            
        # 2. Angular relative offset check (-pi to pi)
        target_angle = np.arctan2(vec[1], vec[0])
        rel_angle = (target_angle - self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # 3. Field of View check
        if abs(rel_angle) > (self.fov_rad / 2.0):
            return False
            
        # 4. USN Perceptual Attenuation:
        # Rel_angle < 0 indicates target is in the LEFT visual field.
        if rel_angle < 0:
            # Perceptual probability drops sharply with USN severity
            left_angle_ratio = abs(rel_angle) / (self.fov_rad / 2.0)
            percept_prob = (1.0 - self.neglect_severity * (0.4 + 0.6 * left_angle_ratio)) * self.attention
            percept_prob = max(0.05, percept_prob)
            return self.rng.random() < percept_prob
        else:
            # Right visual field has normal perception probability
            percept_prob = 0.95 * self.attention
            return self.rng.random() < percept_prob

    def step(self, pet_pos, obstacles=None):
        """
        Advances the patient simulation by one time step DELTA_T.
        Movement vector combines:
        1. Direct attraction to pet if perceived
        2. USN rightward angular bias
        3. Working memory frontier exploration
        4. Obstacle / Wall avoidance
        """
        # Effective walking speed decreases with fatigue
        effective_speed = self.base_speed * (1.0 - 0.4 * self.fatigue)
        step_dist = effective_speed * DELTA_T
        
        # Check if pet is visually perceived
        pet_perceived = self.can_perceive(pet_pos)
        
        if pet_perceived:
            # Target steer angle directly towards pet
            pet_vec = pet_pos - self.pos
            target_angle = np.arctan2(pet_vec[1], pet_vec[0])
            angle_diff = (target_angle - self.heading + np.pi) % (2 * np.pi) - np.pi
            # Smooth heading adjustment towards target
            self.heading += np.clip(angle_diff, -0.4, 0.4)
        else:
            # USN Exploratory Walk:
            # A) USN rightward turning bias (- angle delta in right-hand coordinate system)
            neglect_turn_bias = -0.2 * self.neglect_severity * self.right_bias_weight
            
            # B) Stochastic angular exploration noise
            noise = self.rng.normal(0.0, 0.25)
            
            # C) Frontier Attraction: Seek local unvisited neighbor cells
            frontier_turn = self._get_frontier_turn_bias()
            
            # Update heading with combined cognitive inputs
            self.heading += neglect_turn_bias + noise + 0.3 * frontier_turn
            
        # Ensure heading stays within [-pi, pi]
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # Proposed movement vector
        move_vec = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32) * step_dist
        new_pos = self.pos + move_vec
        
        # Wall Avoidance & Bounding
        margin = 0.4
        if new_pos[0] < margin or new_pos[0] > (ROOM_WIDTH - margin):
            self.heading = np.pi - self.heading  # reflect heading horizontally
            new_pos[0] = np.clip(new_pos[0], margin, ROOM_WIDTH - margin)
            
        if new_pos[1] < margin or new_pos[1] > (ROOM_HEIGHT - margin):
            self.heading = -self.heading  # reflect heading vertically
            new_pos[1] = np.clip(new_pos[1], margin, ROOM_HEIGHT - margin)

        # Obstacle Repulsion & Collision Handling
        if obstacles is not None:
            for obs in obstacles:
                obs_pos = obs['pos']
                obs_radius = obs['radius']
                obs_dist = np.linalg.norm(new_pos - obs_pos)
                if obs_dist < (obs_radius + 0.3):
                    # Repulsion vector away from obstacle
                    repulsion = (new_pos - obs_pos) / (obs_dist + 1e-5)
                    new_pos = obs_pos + repulsion * (obs_radius + 0.35)
                    # Adjust heading away from obstacle
                    self.heading = np.arctan2(repulsion[1], repulsion[0])
                    
        # Apply updated position
        self.pos = np.clip(new_pos, 0.2, 9.8)
        self.trajectory.append(self.pos.copy())
        self._update_visited_cell()
        
        # Build up fatigue slightly over distance walked
        self.fatigue = np.clip(self.fatigue + self.fatigue_rate, 0.0, 1.0)
        
        # Return status dictionary
        is_found = np.linalg.norm(self.pos - pet_pos) <= self.reach_radius
        return {
            "pet_perceived": pet_perceived,
            "pet_found": is_found,
            "position": self.pos.copy(),
            "heading": self.heading,
            "fatigue": self.fatigue
        }

    def _get_frontier_turn_bias(self):
        """Scans adjacent directions to steer towards lesser-visited grid areas."""
        angles = np.linspace(-np.pi/2, np.pi/2, 5)
        cell_w = ROOM_WIDTH / GRID_COLS
        cell_h = ROOM_HEIGHT / GRID_ROWS
        
        min_visits = 9999
        best_angle_offset = 0.0
        
        for offset in angles:
            test_angle = self.heading + offset
            sample_pt = self.pos + np.array([np.cos(test_angle), np.sin(test_angle)]) * 1.2
            c = int(np.clip(sample_pt[0] / cell_w, 0, GRID_COLS - 1))
            r = int(np.clip(sample_pt[1] / cell_h, 0, GRID_ROWS - 1))
            visits = self.visited_grid[r, c]
            if visits < min_visits:
                min_visits = visits
                best_angle_offset = offset
                
        return best_angle_offset
