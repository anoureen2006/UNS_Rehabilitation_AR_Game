"""
unity_dataset.py
Utility script to parse, inspect, and evaluate Unity session logs.
Provides `predict_next_difficulty` to take a raw session JSON string (like sample_unity_session.json)
and output the optimal `difficulty_at_trial` parameters for Unity.
"""

import numpy as np
import json
from typing import Dict, Any, List
from unity_schema import UnitySession, UnityTrial, DifficultyAtTrial
from unity_env import UnityARRehabEnv

def session_to_observation(session: UnitySession) -> np.ndarray:
    """
    Extracts a 15-dimensional state observation vector from a UnitySession instance.
    This vector matches the input format expected by trained RL policies.
    """
    if not session.trials:
        # Default starting observation vector
        return np.array([
            0.0, 0.0, 0.0, 0.0,
            0.3, 10.0/45.0, 1.0/3.0, 3.0/5.0, 1.0,
            0.5, 0.2, 0.0, 1.0, 0.0, 0.0
        ], dtype=np.float32)

    last_trial = session.trials[-1]
    diff = last_trial.difficulty_at_trial

    hit_val = 1.0 if last_trial.hit else 0.0
    rt_norm = np.clip(last_trial.reaction_time_ms / 30000.0, 0.0, 1.0)
    gaze_norm = np.clip(last_trial.gaze_angle_deg / 90.0, 0.0, 1.0)
    hemifield_neg = 1.0 if last_trial.hemifield == "neglected" else 0.0

    spd_norm = np.clip(diff.speed / 1.0, 0.0, 1.0)
    ecc_norm = np.clip(diff.eccentricity_deg / 45.0, 0.0, 1.0)
    dist_norm = np.clip(diff.distance_m / 3.0, 0.0, 1.0)
    cnt_norm = np.clip(diff.target_count / 5.0, 0.0, 1.0)
    tlimit_norm = np.clip(diff.time_limit_s / 30.0, 0.0, 1.0)

    recent_trials = session.trials[-5:]
    rolling_hit = np.mean([1.0 if t.hit else 0.0 for t in recent_trials])
    rolling_rt = np.mean([t.reaction_time_ms for t in recent_trials]) / 30000.0
    
    # Calculate consecutive timeout count
    timeout_cnt = 0
    for t in reversed(session.trials):
        if not t.hit and t.reaction_time_ms >= 30000.0:
            timeout_cnt += 1
        else:
            break
    timeout_norm = np.clip(timeout_cnt / 5.0, 0.0, 1.0)

    neglect_side_left = 1.0 if session.neglect_side == "left" else 0.0
    progress = np.clip(len(session.trials) / 60.0, 0.0, 1.0)
    fatigue_est = np.clip(len(session.trials) * 0.012, 0.0, 1.0)

    return np.array([
        hit_val, rt_norm, gaze_norm, hemifield_neg,
        spd_norm, ecc_norm, dist_norm, cnt_norm, tlimit_norm,
        rolling_hit, rolling_rt, timeout_norm,
        neglect_side_left, progress, fatigue_est
    ], dtype=np.float32)


def predict_next_difficulty(session_json_str: str, model=None) -> Dict[str, Any]:
    """
    Parses a raw Unity session JSON string and uses an RL model (or heuristic rule)
    to return the optimal `difficulty_at_trial` settings for the next Unity trial.
    """
    session = UnitySession.from_json_str(session_json_str)
    obs = session_to_observation(session)
    
    env = UnityARRehabEnv()

    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
        next_diff = env.action_to_difficulty(action)
    else:
        # Heuristic Scaffolding Policy if model is not loaded:
        # Smoothly adapts eccentricity and speed based on rolling hit rate & timeouts
        recent_trials = session.trials[-3:] if session.trials else []
        recent_hits = sum(1 for t in recent_trials if t.hit)
        
        last_diff = session.trials[-1].difficulty_at_trial if session.trials else DifficultyAtTrial()
        
        if recent_hits >= 2:
            # Gradually increase eccentricity by +2.5 degrees (smooth scaffolding)
            new_ecc = float(np.clip(last_diff.eccentricity_deg + 2.5, 5.0, 30.0))
            new_speed = float(np.clip(last_diff.speed + 0.05, 0.2, 0.6))
            new_dist = float(np.clip(last_diff.distance_m + 0.1, 0.8, 2.0))
        elif recent_hits == 0 and len(recent_trials) >= 2:
            # Reduce difficulty on consecutive failures to prevent frustration
            new_ecc = float(np.clip(last_diff.eccentricity_deg - 5.0, 5.0, 30.0))
            new_speed = float(np.clip(last_diff.speed - 0.1, 0.2, 0.6))
            new_dist = float(np.clip(last_diff.distance_m - 0.2, 0.8, 2.0))
        else:
            new_ecc = last_diff.eccentricity_deg
            new_speed = last_diff.speed
            new_dist = last_diff.distance_m

        next_diff = DifficultyAtTrial(
            speed=new_speed,
            eccentricity_deg=new_ecc,
            distance_m=new_dist,
            target_count=3,
            time_limit_s=30.0
        )

    return {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "recommended_next_trial": {
            "difficulty_at_trial": {
                "speed": next_diff.speed,
                "eccentricity_deg": next_diff.eccentricity_deg,
                "distance_m": next_diff.distance_m,
                "target_count": next_diff.target_count,
                "time_limit_s": next_diff.time_limit_s
            }
        }
    }
