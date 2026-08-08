"""
Pydantic models for the USN AR Rehab Game backend.

Coordinate convention:
    All Vector3 fields sent FROM Unity to this backend are expected to be in
    the AR CAMERA'S LOCAL SPACE at the moment they were sampled -- i.e. x is
    left(-)/right(+) relative to where the patient is currently facing,
    z is forward(+) distance, y is up/down. Unity is responsible for this
    conversion (Camera.transform.InverseTransformPoint). This lets the
    backend reason about "left side / right side / eccentricity angle"
    without needing to know anything about the room's world coordinates.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class NeglectSide(str, Enum):
    left = "left"
    right = "right"


class ExerciseMode(str, Enum):
    bird_chase = "bird_chase"
    star_collect = "star_collect"


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class DifficultyParams(BaseModel):
    speed: float = Field(..., description="Target movement speed, arbitrary units, higher = harder")
    eccentricity_deg: float = Field(..., description="Target angle off-center the patient must turn to reach")
    distance_m: float = Field(..., description="Distance from patient to target spawn point, meters")
    target_count: int = Field(default=1, description="Used in star_collect mode; ignored in bird_chase")
    time_limit_s: float = Field(default=30.0, description="Time budget for the trial/round")


class SessionStartRequest(BaseModel):
    patient_id: str
    neglect_side: NeglectSide
    exercise_mode: ExerciseMode = ExerciseMode.bird_chase


class SessionStartResponse(BaseModel):
    session_id: str
    initial_difficulty: DifficultyParams


class TrialResult(BaseModel):
    hit: bool
    reaction_time_ms: Optional[float] = None
    gaze_angle_deg: Optional[float] = None
    target_position: Optional[Vector3] = None


class NextTargetRequest(BaseModel):
    candidate_points: List[Vector3] = Field(default_factory=list)
    last_result: Optional[TrialResult] = None


class NextTargetResponse(BaseModel):
    spawn_point: Vector3
    difficulty: DifficultyParams


class NextRoundRequest(BaseModel):
    """Used for star_collect mode: request a whole batch of targets at once
    instead of one at a time. last_round_results carries every star's
    outcome from the previous round (empty list on the very first call)."""
    candidate_points: List[Vector3] = Field(default_factory=list)
    last_round_results: List[TrialResult] = Field(default_factory=list)


class NextRoundResponse(BaseModel):
    spawn_points: List[Vector3]
    difficulty: DifficultyParams


class SessionMetricsResponse(BaseModel):
    session_id: str
    total_trials: int
    hit_rate_overall: float
    hit_rate_neglected_side: float
    hit_rate_non_neglected_side: float
    avg_reaction_time_ms_overall: Optional[float]
    avg_reaction_time_ms_neglected_side: Optional[float]
    asymmetry_index: Optional[float] = Field(
        None,
        description=(
            "(-1 to 1). Negative = more misses/slower on neglected side. "
            "0 = perfectly symmetric performance. Digital analog of "
            "line-bisection / Catherine Bergego Scale asymmetry."
        ),
    )
