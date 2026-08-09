"""
Simulates a full Unity client session against the running backend, without
needing a phone/headset. Useful for:
    - Testing the difficulty controller's escalation/de-escalation over many
      trials quickly
    - Generating sample data for the CSV export / metrics endpoints before
      device testing is smooth
    - Isolating "is this a backend bug or a Unity bug" when something breaks

Run with (backend must already be running separately via uvicorn):
    pip install requests --break-system-packages
    python simulate_client.py --trials 50 --neglect-side left --hit-rate 0.7
"""

from __future__ import annotations
import argparse
import random
import time
import requests


def simulate(base_url: str, patient_id: str, neglect_side: str, exercise_mode: str,
             num_trials: int, hit_rate: float):
    r = requests.post(f"{base_url}/session/start", json={
        "patient_id": patient_id,
        "neglect_side": neglect_side,
        "exercise_mode": exercise_mode,
    })
    r.raise_for_status()
    data = r.json()
    session_id = data["session_id"]
    difficulty = data["initial_difficulty"]
    print(f"Session started: {session_id}")
    print(f"Initial difficulty: {difficulty}")

    last_result = None
    for i in range(num_trials):
        # Fake "candidate points" the way SafeSpawnPointFinder would send
        # them: a handful of points scattered left/right of center in
        # camera-local space.
        candidates = []
        for _ in range(6):
            x = random.uniform(-2.0, 2.0)
            z = random.uniform(1.0, 3.0)
            candidates.append({"x": round(x, 2), "y": 0.0, "z": round(z, 2)})

        payload = {"candidate_points": candidates, "last_result": last_result}
        r = requests.post(f"{base_url}/session/{session_id}/next-target", json=payload)
        r.raise_for_status()
        resp = r.json()
        spawn_point = resp["spawn_point"]
        difficulty = resp["difficulty"]

        hit = random.random() < hit_rate
        reaction_time = random.uniform(400, 1200) if hit else difficulty["time_limit_s"] * 1000

        last_result = {
            "hit": hit,
            "reaction_time_ms": round(reaction_time, 1),
            "gaze_angle_deg": round(random.uniform(0, difficulty["eccentricity_deg"]), 1),
            "target_position": spawn_point,
        }

        print(f"trial {i+1:3d} | hit={hit!s:5} | speed={difficulty['speed']:.2f} "
              f"| ecc={difficulty['eccentricity_deg']:.0f} deg | dist={difficulty['distance_m']:.2f}m "
              f"| spawn=({spawn_point['x']:.2f},{spawn_point['z']:.2f})")

        time.sleep(0.05)  # don't hammer the server instantly, mimics real pacing loosely

    # Submit the final trial's result too (normally the NEXT next-target call
    # would carry it, but the loop above ends before that call happens)
    requests.post(f"{base_url}/session/{session_id}/next-target",
                   json={"candidate_points": [], "last_result": last_result})

    r = requests.get(f"{base_url}/session/{session_id}/metrics")
    print("\nFinal metrics:")
    print(r.json())
    print(f"\nCSV export: {base_url}/session/{session_id}/export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--patient-id", default="sim01")
    parser.add_argument("--neglect-side", choices=["left", "right"], default="left")
    parser.add_argument("--exercise-mode", default="bird_chase")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hit-rate", type=float, default=0.7, help="0.0-1.0 probability of a simulated hit")
    args = parser.parse_args()

    simulate(args.base_url, args.patient_id, args.neglect_side, args.exercise_mode,
              args.trials, args.hit_rate)
