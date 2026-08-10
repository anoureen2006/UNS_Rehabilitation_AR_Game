"""
test_unity_integration.py
Tests parsing Unity session JSON, building the observation vector,
and serving RL difficulty predictions.
"""

import unittest
import json
import os
import numpy as np

from unity_schema import UnitySession, UnityTrial, DifficultyAtTrial
from unity_env import UnityARRehabEnv
from unity_dataset import session_to_observation, predict_next_difficulty

class TestUnityARIntegration(unittest.TestCase):

    def test_parse_sample_unity_json(self):
        sample_path = "sample_unity_session.json"
        self.assertTrue(os.path.exists(sample_path))
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
            
        session = UnitySession.from_json_str(json_str)
        self.assertEqual(session.session_id, "fa8a4ce9")
        self.assertEqual(session.patient_id, "demo01")
        self.assertEqual(session.neglect_side, "left")
        self.assertEqual(session.exercise_mode, "bird_chase")
        self.assertGreaterEqual(len(session.trials), 10)

    def test_extract_observation_vector(self):
        with open("sample_unity_session.json", 'r', encoding='utf-8') as f:
            json_str = f.read()
        session = UnitySession.from_json_str(json_str)
        
        obs = session_to_observation(session)
        self.assertEqual(obs.shape, (15,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_predict_next_difficulty(self):
        with open("sample_unity_session.json", 'r', encoding='utf-8') as f:
            json_str = f.read()
            
        prediction = predict_next_difficulty(json_str)
        self.assertEqual(prediction["session_id"], "fa8a4ce9")
        self.assertEqual(prediction["patient_id"], "demo01")
        
        next_diff = prediction["recommended_next_trial"]["difficulty_at_trial"]
        self.assertIn("speed", next_diff)
        self.assertIn("eccentricity_deg", next_diff)
        self.assertIn("distance_m", next_diff)
        self.assertGreaterEqual(next_diff["speed"], 0.2)
        self.assertLessEqual(next_diff["speed"], 0.8)

if __name__ == '__main__':
    unittest.main()
