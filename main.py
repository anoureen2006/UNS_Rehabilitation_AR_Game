"""
USN AR Rehab Game -- backend entrypoint.

Run locally with:
    pip install -r requirements.txt --break-system-packages
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then point Unity's NetworkClient base URL at:
    http://<your-laptop-LAN-IP>:8000
(not "localhost" -- a physical device on Wi-Fi can't resolve your laptop's
localhost. Find your LAN IP with `ipconfig` on Windows or `ifconfig`/`ip a`
on Mac/Linux, and make sure the phone/headset is on the SAME Wi-Fi network.)
"""

from __future__ import annotations
import csv
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from schemas import (
    SessionStartRequest,
    SessionStartResponse,
    NextTargetRequest,
    NextTargetResponse,
    NextRoundRequest,
    NextRoundResponse,
    SessionMetricsResponse,
    DifficultyParams,
)
from session_store import store
import difficulty_controller as dc
import spawn_filter

app = FastAPI(title="USN AR Rehab Game Backend")

# Wide open for hackathon speed. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest):
    initial = dc.initial_difficulty()
    session = store.create_session(
        patient_id=req.patient_id,
        neglect_side=req.neglect_side,
        exercise_mode=req.exercise_mode,
        initial_difficulty=initial,
    )
    return SessionStartResponse(session_id=session.session_id, initial_difficulty=initial)


@app.post("/session/{session_id}/next-target", response_model=NextTargetResponse)
def next_target(session_id: str, req: NextTargetRequest):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    # 1. Log the previous trial's result, if any (first call after /start has none)
    if req.last_result is not None:
        store.record_trial(session, req.last_result)
        session.difficulty = dc.next_difficulty(
            session.difficulty,
            session.consecutive_hits,
            session.consecutive_misses,
        )

    # 2. Pick where the next target should spawn from Unity's candidate points
    spawn_point = spawn_filter.choose_spawn_point(
        req.candidate_points,
        session.difficulty,
        session.neglect_side,
    )

    return NextTargetResponse(spawn_point=spawn_point, difficulty=session.difficulty)


@app.post("/session/{session_id}/next-round", response_model=NextRoundResponse)
def next_round(session_id: str, req: NextRoundRequest):
    """
    star_collect mode equivalent of /next-target: instead of one target at a
    time, spawns a whole batch (difficulty.target_count) at once. Call this
    once at the start of a round, then again after the patient has
    collected/timed-out on all stars in the current round, passing every
    star's result from that round in last_round_results.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    if req.last_round_results:
        for result in req.last_round_results:
            store.record_trial(session, result)

        hits = sum(1 for r in req.last_round_results if r.hit)
        round_hit_rate = hits / len(req.last_round_results)
        session.difficulty = dc.next_difficulty_for_round(session.difficulty, round_hit_rate)

    spawn_points = spawn_filter.choose_multiple_spawn_points(
        req.candidate_points,
        session.difficulty,
        session.neglect_side,
    )

    return NextRoundResponse(spawn_points=spawn_points, difficulty=session.difficulty)


@app.get("/session/{session_id}/metrics", response_model=SessionMetricsResponse)
def get_metrics(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    trials = session.trials
    total = len(trials)
    if total == 0:
        return SessionMetricsResponse(
            session_id=session_id,
            total_trials=0,
            hit_rate_overall=0.0,
            hit_rate_neglected_side=0.0,
            hit_rate_non_neglected_side=0.0,
            avg_reaction_time_ms_overall=None,
            avg_reaction_time_ms_neglected_side=None,
            asymmetry_index=None,
        )

    def hit_rate(subset):
        return sum(1 for t in subset if t.hit) / len(subset) if subset else 0.0

    def avg_rt(subset):
        rts = [t.reaction_time_ms for t in subset if t.reaction_time_ms is not None]
        return sum(rts) / len(rts) if rts else None

    neglected = [t for t in trials if t.hemifield == "neglected"]
    non_neglected = [t for t in trials if t.hemifield == "non_neglected"]

    hr_neg = hit_rate(neglected)
    hr_non = hit_rate(non_neglected)
    asymmetry = None
    if neglected and non_neglected:
        # -1 = fully asymmetric against neglected side, 0 = symmetric
        asymmetry = round(hr_neg - hr_non, 3)

    return SessionMetricsResponse(
        session_id=session_id,
        total_trials=total,
        hit_rate_overall=round(hit_rate(trials), 3),
        hit_rate_neglected_side=round(hr_neg, 3),
        hit_rate_non_neglected_side=round(hr_non, 3),
        avg_reaction_time_ms_overall=avg_rt(trials),
        avg_reaction_time_ms_neglected_side=avg_rt(neglected),
        asymmetry_index=asymmetry,
    )


@app.get("/session/{session_id}/export")
def export_session_csv(session_id: str):
    """
    Returns the full trial-by-trial log as CSV -- useful for showing an
    expert reviewer real numbers, and as raw data for any later analysis.
    Open the URL directly in a browser to download/view it.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "trial_number", "timestamp", "hit", "reaction_time_ms",
        "gaze_angle_deg", "hemifield", "speed", "eccentricity_deg", "distance_m",
    ])
    for i, t in enumerate(session.trials, start=1):
        diff = t.difficulty_at_trial
        writer.writerow([
            i, t.timestamp, t.hit, t.reaction_time_ms, t.gaze_angle_deg,
            t.hemifield, diff.get("speed"), diff.get("eccentricity_deg"), diff.get("distance_m"),
        ])

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"},
    )
