"""
reward.py
Rehabilitation reward function module for the AR Digital Pet RL System.
Translates clinical goals (spatial neglect recovery, left-side room coverage, patient engagement)
into mathematical reward shaping terms for RL training.
"""

import numpy as np
from config import REWARD_CONFIG, GRID_COLS, GRID_ROWS, LEFT_BOUND, EXTREME_LEFT_BOUND

class RehabilitationRewardCalculator:
    """
    Computes reward signal for digital pet placement and patient trajectory evaluation.
    Encourages placing the pet such that the patient naturally scans and traverses
    their neglected left field before reaching the goal.
    """
    def __init__(self, config=None):
        self.cfg = config if config is not None else REWARD_CONFIG

    def compute_step_reward(self, patient_state, pet_pos, prev_visited_grid, step_count, max_steps):
        """
        Computes step-level and terminal rewards based on current patient progress.
        
        Parameters:
            patient_state (dict): Output from PatientSim step containing position, heading, fatigue, pet_found.
            pet_pos (np.ndarray): [x, y] position of digital pet.
            prev_visited_grid (np.ndarray): 20x20 visited cell counts before this step.
            step_count (int): Current episode step number.
            max_steps (int): Max allowed episode steps.
            
        Returns:
            total_reward (float): Combined reward scalar.
            info (dict): Breakdown of individual reward components for analysis.
        """
        pos = patient_state["position"]
        is_found = patient_state["pet_found"]
        curr_grid = patient_state["visited_grid"]
        neglect_severity = patient_state["neglect_severity"]
        
        # Calculate cell coverage metrics
        total_cells = GRID_ROWS * GRID_COLS
        visited_mask = curr_grid > 0
        prev_visited_mask = prev_visited_grid > 0
        cell_w = 10.0 / GRID_COLS
        
        new_cells_discovered = np.sum(visited_mask & ~prev_visited_mask)
        
        # Split room into Left (cols 0..9) and Right (cols 10..19)
        left_cols = GRID_COLS // 2
        left_visited_mask = visited_mask[:, :left_cols]
        right_visited_mask = visited_mask[:, left_cols:]
        
        left_expl_ratio = np.mean(left_visited_mask)
        right_expl_ratio = np.mean(right_visited_mask)
        total_expl_ratio = np.mean(visited_mask)
        
        # Coverage Balance (0 = perfectly balanced or no exploration, 1 = maximum asymmetry)
        expl_asymmetry = abs(left_expl_ratio - right_expl_ratio) / (left_expl_ratio + right_expl_ratio + 1e-6)
        
        # 1. New Cell Discovery Reward (scaled up if cell is on the neglected left side)
        r_new_cells = 0.0
        if new_cells_discovered > 0:
            # Check if newly discovered cell is on the left
            col = int(np.clip(pos[0] / cell_w, 0, GRID_COLS - 1))
            if col < left_cols:
                # Left side cell discovery
                is_extreme_left = pos[0] < EXTREME_LEFT_BOUND
                left_mult = self.cfg["left_side_multiplier"] * (1.0 + 0.5 * neglect_severity)
                r_new_cells = self.cfg["new_cell_visited"] * new_cells_discovered * left_mult
                if is_extreme_left:
                    r_new_cells += self.cfg["extreme_left_bonus"]
            else:
                # Right side cell discovery
                r_new_cells = self.cfg["new_cell_visited"] * new_cells_discovered * 0.5
                
        # 2. Neglected Region Reach Bonus (one-time check for entering deep left)
        r_neglect_milestone = 0.0
        if pos[0] < EXTREME_LEFT_BOUND and not np.any(prev_visited_grid[:, :int(EXTREME_LEFT_BOUND / cell_w)] > 0):
            r_neglect_milestone = 5.0 * (1.0 + neglect_severity)

        # 3. Penalties (Standing Still, Rightward Bias, Step Penalty)
        dist_moved = np.linalg.norm(pos - patient_state["prev_pos"]) if "prev_pos" in patient_state else 0.1
        p_stagnation = self.cfg["standing_still_penalty"] if dist_moved < 0.05 else 0.0
        
        # Rightward movement penalty if left exploration is severely lagging
        p_right_bias = 0.0
        if pos[0] > LEFT_BOUND and left_expl_ratio < 0.15 and step_count > 15:
            p_right_bias = self.cfg["right_bias_penalty"]
            
        r_step = self.cfg["step_penalty"]
        
        # 4. Terminal Outcome Rewards (Pet Found vs Timeout)
        r_terminal = 0.0
        p_premature = 0.0
        
        if is_found:
            if step_count < 10 or left_expl_ratio < 0.15:
                # Premature discovery penalty: pet was placed too easy / patient didn't explore
                p_premature = self.cfg["premature_found_penalty"]
            else:
                # Clinical Success: Pet found after significant left exploration
                base_pet_bonus = self.cfg["pet_found_base"]
                rehab_bonus = self.cfg["rehab_success_left_scale"] * left_expl_ratio
                balance_bonus = 5.0 * (1.0 - expl_asymmetry)
                r_terminal = base_pet_bonus + rehab_bonus + balance_bonus
        elif step_count >= max_steps:
            # Timeout penalty: pet was placed too far or unreachable
            r_terminal = self.cfg["timeout_penalty"]

        # Sum total step reward
        total_reward = (r_new_cells + r_neglect_milestone + r_step +
                        p_stagnation + p_right_bias + p_premature + r_terminal)
        
        info = {
            "r_new_cells": float(r_new_cells),
            "r_neglect_milestone": float(r_neglect_milestone),
            "p_stagnation": float(p_stagnation),
            "p_right_bias": float(p_right_bias),
            "p_premature": float(p_premature),
            "r_terminal": float(r_terminal),
            "left_expl_ratio": float(left_expl_ratio),
            "right_expl_ratio": float(right_expl_ratio),
            "total_expl_ratio": float(total_expl_ratio),
            "expl_asymmetry": float(expl_asymmetry),
            "step_count": step_count
        }
        
        return total_reward, info
