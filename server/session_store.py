"""
Simple session store. In-memory dict for speed during the hackathon, with
every trial also appended to a JSON file on disk so nothing is lost if the
server restarts and so the raw data is available for the expert-review demo
and any later metrics analysis.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from schemas import DifficultyParams, NeglectSide, ExerciseMode, TrialResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class TrialRecord:
    timestamp: float
    hit: bool
    reaction_time_ms: Optional[float]
    gaze_angle_deg: Optional[float]
    hemifield: Optional[str]  # "neglected" or "non_neglected"
    difficulty_at_trial: dict


@dataclass
class Session:
    session_id: str
    patient_id: str
    neglect_side: NeglectSide
    exercise_mode: ExerciseMode
    difficulty: DifficultyParams
    created_at: float = field(default_factory=time.time)
    trials: List[TrialRecord] = field(default_factory=list)
    consecutive_hits: int = 0
    consecutive_misses: int = 0


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_session(
        self,
        patient_id: str,
        neglect_side: NeglectSide,
        exercise_mode: ExerciseMode,
        initial_difficulty: DifficultyParams,
    ) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            patient_id=patient_id,
            neglect_side=neglect_side,
            exercise_mode=exercise_mode,
            difficulty=initial_difficulty,
        )
        self._sessions[session_id] = session
        self._persist_session_meta(session)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def record_trial(self, session: Session, result: TrialResult) -> None:
        hemifield = None
        if result.target_position is not None:
            # x < 0 => left of patient's facing direction, x > 0 => right
            is_left = result.target_position.x < 0
            neglected_is_left = session.neglect_side == NeglectSide.left
            hemifield = "neglected" if (is_left == neglected_is_left) else "non_neglected"

        record = TrialRecord(
            timestamp=time.time(),
            hit=result.hit,
            reaction_time_ms=result.reaction_time_ms,
            gaze_angle_deg=result.gaze_angle_deg,
            hemifield=hemifield,
            difficulty_at_trial=session.difficulty.model_dump(),
        )
        session.trials.append(record)

        if result.hit:
            session.consecutive_hits += 1
            session.consecutive_misses = 0
        else:
            session.consecutive_misses += 1
            session.consecutive_hits = 0

        self._append_trial_to_disk(session, record)

    def _persist_session_meta(self, session: Session) -> None:
        path = os.path.join(DATA_DIR, f"{session.session_id}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "session_id": session.session_id,
                    "patient_id": session.patient_id,
                    "neglect_side": session.neglect_side.value,
                    "exercise_mode": session.exercise_mode.value,
                    "created_at": session.created_at,
                    "trials": [],
                },
                f,
                indent=2,
            )

    def _append_trial_to_disk(self, session: Session, record: TrialRecord) -> None:
        path = os.path.join(DATA_DIR, f"{session.session_id}.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["trials"].append(asdict(record))
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Single shared instance imported by main.py
store = SessionStore()
