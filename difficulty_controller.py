"""
Day 1 adaptive difficulty controller: rule-based, not ML.

This intentionally exposes the SAME function signature you'd use for a real
RL policy later (session_state -> DifficultyParams), so swapping in an
ML-Agents or bandit-based policy on Day 3+ means changing only this file's
internals, not any caller.

Logic:
    - 3 consecutive hits  -> increase difficulty (staircase up)
    - 2 consecutive misses -> decrease difficulty (staircase down)
    - Values are clamped to safe/sane bounds
    - Neglected-side eccentricity is nudged higher over time; non-neglected
      side is intentionally kept easier so the training pressure stays where
      it clinically belongs
"""

from __future__ import annotations
from schemas import DifficultyParams

# Bounds -- tune during Day 2 playtesting
SPEED_MIN, SPEED_MAX = 0.3, 2.0
ECC_MIN, ECC_MAX = 10.0, 80.0  # degrees off-center
DIST_MIN, DIST_MAX = 1.0, 3.0  # meters

SPEED_STEP = 0.1
ECC_STEP_DEG = 5.0
DIST_STEP_M = 0.1
TARGET_COUNT_MIN, TARGET_COUNT_MAX = 3, 8


def initial_difficulty() -> DifficultyParams:
    return DifficultyParams(
        speed=0.5,
        eccentricity_deg=20.0,
        distance_m=1.5,
        target_count=3,
        time_limit_s=30.0,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def next_difficulty(
    current: DifficultyParams,
    consecutive_hits: int,
    consecutive_misses: int,
) -> DifficultyParams:
    speed = current.speed
    ecc = current.eccentricity_deg
    dist = current.distance_m

    if consecutive_hits >= 3:
        speed = _clamp(speed + SPEED_STEP, SPEED_MIN, SPEED_MAX)
        ecc = _clamp(ecc + ECC_STEP_DEG, ECC_MIN, ECC_MAX)
        dist = _clamp(dist + DIST_STEP_M, DIST_MIN, DIST_MAX)
    elif consecutive_misses >= 2:
        speed = _clamp(speed - SPEED_STEP, SPEED_MIN, SPEED_MAX)
        ecc = _clamp(ecc - ECC_STEP_DEG, ECC_MIN, ECC_MAX)
        dist = _clamp(dist - DIST_STEP_M, DIST_MIN, DIST_MAX)
    # else: no change this trial, hold steady

    return DifficultyParams(
        speed=round(speed, 3),
        eccentricity_deg=round(ecc, 2),
        distance_m=round(dist, 2),
        target_count=current.target_count,
        time_limit_s=current.time_limit_s,
    )


def next_difficulty_for_round(
    current: DifficultyParams,
    round_hit_rate: float,
) -> DifficultyParams:
    """
    star_collect mode has no single "consecutive streak" -- a round spawns
    several stars at once and they're all resolved together. Instead we
    look at what fraction of that round's stars were collected:
        - hit_rate >= 0.8  -> harder next round (more stars, wider spread)
        - hit_rate <= 0.4  -> easier next round
        - otherwise        -> hold steady
    Same eccentricity/distance/speed bounds as bird_chase apply, since
    eccentricity still controls how far toward the neglected side stars
    get placed, and speed still affects any subtle idle motion.
    """
    speed = current.speed
    ecc = current.eccentricity_deg
    dist = current.distance_m
    count = current.target_count

    if round_hit_rate >= 0.8:
        ecc = _clamp(ecc + ECC_STEP_DEG, ECC_MIN, ECC_MAX)
        dist = _clamp(dist + DIST_STEP_M, DIST_MIN, DIST_MAX)
        count = int(_clamp(count + 1, TARGET_COUNT_MIN, TARGET_COUNT_MAX))
    elif round_hit_rate <= 0.4:
        ecc = _clamp(ecc - ECC_STEP_DEG, ECC_MIN, ECC_MAX)
        dist = _clamp(dist - DIST_STEP_M, DIST_MIN, DIST_MAX)
        count = int(_clamp(count - 1, TARGET_COUNT_MIN, TARGET_COUNT_MAX))

    return DifficultyParams(
        speed=round(speed, 3),
        eccentricity_deg=round(ecc, 2),
        distance_m=round(dist, 2),
        target_count=count,
        time_limit_s=current.time_limit_s,
    )
