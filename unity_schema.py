"""
unity_schema.py
Data structures and parser for the exact Unity AR Rehabilitation session log JSON format.
Directly parses session_id, patient_id, neglect_side, exercise_mode, trials list,
and trial difficulty settings.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class DifficultyAtTrial:
    speed: float = 0.3
    eccentricity_deg: float = 10.0
    distance_m: float = 1.0
    target_count: int = 3
    time_limit_s: float = 30.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DifficultyAtTrial":
        return cls(
            speed=float(d.get("speed", 0.3)),
            eccentricity_deg=float(d.get("eccentricity_deg", 10.0)),
            distance_m=float(d.get("distance_m", 1.0)),
            target_count=int(d.get("target_count", 3)),
            time_limit_s=float(d.get("time_limit_s", 30.0))
        )

@dataclass
class UnityTrial:
    timestamp: float
    hit: bool
    reaction_time_ms: float
    gaze_angle_deg: float
    hemifield: str  # "neglected" or "non_neglected"
    difficulty_at_trial: DifficultyAtTrial

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnityTrial":
        diff_dict = d.get("difficulty_at_trial", {})
        return cls(
            timestamp=float(d.get("timestamp", 0.0)),
            hit=bool(d.get("hit", False)),
            reaction_time_ms=float(d.get("reaction_time_ms", 0.0)),
            gaze_angle_deg=float(d.get("gaze_angle_deg", 0.0)),
            hemifield=str(d.get("hemifield", "non_neglected")),
            difficulty_at_trial=DifficultyAtTrial.from_dict(diff_dict)
        )

@dataclass
class UnitySession:
    session_id: str
    patient_id: str
    neglect_side: str
    exercise_mode: str
    created_at: float
    trials: List[UnityTrial] = field(default_factory=list)

    @classmethod
    def from_json_str(cls, json_str: str) -> "UnitySession":
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitySession":
        trials_list = [UnityTrial.from_dict(t) for t in d.get("trials", [])]
        return cls(
            session_id=str(d.get("session_id", "unknown")),
            patient_id=str(d.get("patient_id", "demo01")),
            neglect_side=str(d.get("neglect_side", "left")),
            exercise_mode=str(d.get("exercise_mode", "bird_chase")),
            created_at=float(d.get("created_at", 0.0)),
            trials=trials_list
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
