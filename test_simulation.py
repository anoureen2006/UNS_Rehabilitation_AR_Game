"""
test_simulation.py
Unit tests verifying the AR Rehabilitation RL environment, patient USN movement model,
pet placement mechanics, and reward signal behavior.
"""

import unittest
import numpy as np
import os

from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, NUM_CELLS
from environment import ARRehabEnv
from patient import PatientSim
from digital_pet import DigitalPet
from reward import RehabilitationRewardCalculator

class TestARRehabSimulation(unittest.TestCase):

    def test_environment_reset_and_observation_shape(self):
        env = ARRehabEnv(neglect_severity=0.7, seed=42)
        obs, info = env.reset()
        
        self.assertEqual(obs.shape, (44,))
        self.assertGreaterEqual(info["num_obstacles"], 6)
        self.assertLessEqual(info["num_obstacles"], 12)
        self.assertGreaterEqual(obs[0], 0.0) # normalized patient position x
        self.assertLessEqual(obs[0], 1.0)

    def test_patient_usn_turn_bias(self):
        """Verifies that high USN severity induces systematic rightward turn bias."""
        patient_high_neglect = PatientSim(init_pos=[5.0, 5.0], neglect_severity=0.9, seed=42)
        pet_pos_far_left = np.array([0.5, 5.0], dtype=np.float32)
        
        initial_heading = patient_high_neglect.heading
        # Run 10 steps without pet perception
        right_turns = 0
        for _ in range(10):
            prev_h = patient_high_neglect.heading
            patient_high_neglect.step(pet_pos_far_left)
            # Negative angular delta represents rightward turn in standard coordinate system
            if (patient_high_neglect.heading - prev_h) < 0:
                right_turns += 1

        self.assertGreaterEqual(right_turns, 5)

    def test_digital_pet_cell_mapping(self):
        pet = DigitalPet(init_cell=0) # Top-left cell
        np.testing.assert_allclose(pet.pos, [0.25, 0.25], atol=1e-2)
        
        pet.set_position(399) # Bottom-right cell (row 19, col 19)
        np.testing.assert_allclose(pet.pos, [9.75, 9.75], atol=1e-2)
        self.assertEqual(pet.cell_id, 399)

    def test_reward_shaping(self):
        calc = RehabilitationRewardCalculator()
        patient_state = {
            "position": np.array([1.5, 5.0], dtype=np.float32),
            "prev_pos": np.array([3.0, 5.0], dtype=np.float32),
            "pet_found": False,
            "visited_grid": np.ones((20, 20), dtype=np.int32),
            "neglect_severity": 0.8
        }
        prev_visited = np.zeros((20, 20), dtype=np.int32)
        
        reward, info = calc.compute_step_reward(patient_state, np.array([1.5, 5.0]), prev_visited, step_count=20, max_steps=200)
        self.assertGreater(reward, 0.0) # Should award high positive reward for discovering extreme left cells
        self.assertGreater(info["r_new_cells"], 0.0)

if __name__ == '__main__':
    unittest.main()
