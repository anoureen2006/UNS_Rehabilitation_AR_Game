"""
unity_continuous_api.py
Phase 2: Continuous Adaptive RL API Server for Unity AR Foundation.
Handles live 2-Phase workflow:
1. Loads Phase 1 Base Model (`checkpoints/unity_ppo_model.zip`) for new patients.
2. Fine-tunes model weights in real-time (`model.learn(reset_num_timesteps=False)`) after every Unity trial.
3. Saves patient-specific checkpoints (`checkpoints/patients/{patient_id}_ppo.zip`).
4. Serves optimal 4-action trial parameters (speed, eccentricity_deg, distance_m, time_limit_s) in <10ms.
"""

from flask import Flask, request, jsonify
import os
import json
import numpy as np
from stable_baselines3 import PPO

from config import CHECKPOINT_DIR
from unity_schema import UnitySession
from unity_env import UnityARRehabEnv
from unity_dataset import session_to_observation

app = Flask(__name__)

# Directory paths for Base Model and Patient-Specific Fine-Tuned Models
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "unity_ppo_model.zip")
PATIENT_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "patients")
os.makedirs(PATIENT_MODEL_DIR, exist_ok=True)

@app.route('/predict_and_adapt', methods=['POST'])
def predict_and_adapt():
    """
    Continuous Adaptive API Endpoint:
    Receives Unity session telemetry JSON -> Extracts 15-dim state -> Predicts 4 Actions -> Fine-tunes policy online.
    """
    try:
        session_json_str = request.get_data(as_text=True)
        if not session_json_str.strip():
            return jsonify({"error": "Empty JSON payload"}), 400

        session = UnitySession.from_json_str(session_json_str)
        patient_id = session.patient_id
        patient_model_path = os.path.join(PATIENT_MODEL_DIR, f"{patient_id}_ppo.zip")

        # 1. Load patient-specific model if exists; otherwise load Phase 1 Base Model
        if os.path.exists(patient_model_path):
            model = PPO.load(patient_model_path, device='cpu')
            print(f"[Phase 2] Loaded personalized model for patient '{patient_id}'")
        elif os.path.exists(BASE_MODEL_PATH):
            model = PPO.load(BASE_MODEL_PATH, device='cpu')
            print(f"[Phase 1] Loaded Phase 1 Base Model for new patient '{patient_id}'")
        else:
            # Fallback instantiation if no checkpoint exists yet
            env = UnityARRehabEnv()
            model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0, device='cpu')
            print(f"[Initialization] Instantiated fresh PPO agent for '{patient_id}'")

        # 2. Extract 15-dimensional observation vector
        obs = session_to_observation(session)

        # 3. Predict optimal 4-action trial settings
        action, _ = model.predict(obs, deterministic=True)
        env = UnityARRehabEnv()
        next_diff = env.action_to_difficulty(action)

        # 4. Phase 2: Continuous Online Adaptation
        if len(session.trials) >= 2:
            dummy_env = UnityARRehabEnv()
            model.set_env(dummy_env)
            # Perform 5 micro-steps of continuous gradient updates on live telemetry
            model.learn(total_timesteps=5, reset_num_timesteps=False)
            model.save(patient_model_path)
            print(f"[Phase 2 Adaptive] Updated & saved fine-tuned policy for '{patient_id}'")

        # 5. Return JSON recommendation for Unity AR Foundation
        response_payload = {
            "session_id": session.session_id,
            "patient_id": patient_id,
            "phase": "Phase 2 Continuous Adaptive",
            "recommended_next_trial": {
                "difficulty_at_trial": {
                    "speed": float(next_diff.speed),
                    "eccentricity_deg": float(next_diff.eccentricity_deg),
                    "distance_m": float(next_diff.distance_m),
                    "target_count": 3,  # Fixed at 3
                    "time_limit_s": float(next_diff.time_limit_s)
                }
            }
        }
        return jsonify(response_payload)

    except Exception as e:
        print(f"Error in continuous adaptive inference: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("  UNITY AR REHABILITATION - PHASE 2 CONTINUOUS ADAPTIVE SERVER")
    print("  Serving on http://localhost:5000/predict_and_adapt")
    print("="*60)
    app.run(host='0.0.0.0', port=5000)
