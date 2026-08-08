"""
Day 1 "vision" module. NOT YOLO/segmentation yet -- that's a Day 3+ stretch
goal layered on top of this same function signature. For Day 1, "safety" and
"relevance" are approximated with simple geometry:

    - Points are expected in the AR camera's LOCAL space (see schemas.py note)
    - We score each candidate by:
        1. How close its angle is to the current target eccentricity
        2. Whether it's on the correct (neglected) side, weighted by a bias
        3. How close its distance is to the current target distance

The highest-scoring candidate is returned as the spawn point. If Unity sends
no usable candidates, we fall back to a synthesized point at the target
eccentricity/distance so the game never stalls.
"""

from __future__ import annotations
import math
from typing import List, Optional

from schemas import Vector3, DifficultyParams, NeglectSide

# How strongly we prefer the neglected side over the non-neglected side.
# 0.7 means ~70% weight given to matching the correct side.
NEGLECT_SIDE_BIAS = 0.7


def _angle_deg(point: Vector3) -> float:
    """Angle off-center (forward = +z), signed: negative = left, positive = right."""
    return math.degrees(math.atan2(point.x, max(point.z, 1e-4)))


def _distance_m(point: Vector3) -> float:
    return math.sqrt(point.x**2 + point.y**2 + point.z**2)


def _score_point(point: Vector3, difficulty: DifficultyParams, neglect_side: NeglectSide) -> float:
    angle = _angle_deg(point)
    dist = _distance_m(point)

    target_angle = difficulty.eccentricity_deg if neglect_side == NeglectSide.right else -difficulty.eccentricity_deg
    angle_error = abs(angle - target_angle) / 90.0  # normalize roughly to 0-1
    dist_error = abs(dist - difficulty.distance_m) / max(difficulty.distance_m, 0.1)

    is_correct_side = (angle < 0) == (neglect_side == NeglectSide.left)
    side_penalty = 0.0 if is_correct_side else NEGLECT_SIDE_BIAS

    # Lower is better; combine into one score
    return angle_error + dist_error + side_penalty


def choose_spawn_point(
    candidates: List[Vector3],
    difficulty: DifficultyParams,
    neglect_side: NeglectSide,
) -> Vector3:
    if not candidates:
        return _synthesize_fallback_point(difficulty, neglect_side)

    best = min(candidates, key=lambda p: _score_point(p, difficulty, neglect_side))
    return best


def choose_multiple_spawn_points(
    candidates: List[Vector3],
    difficulty: DifficultyParams,
    neglect_side: NeglectSide,
) -> List[Vector3]:
    """
    For star_collect mode: pick difficulty.target_count points, biased
    toward the neglected side. Roughly 70% of stars go on the neglected
    side, 30% on the other -- matching NEGLECT_SIDE_BIAS -- so the exercise
    still requires some full-field scanning, not just a fixed stare toward
    one side.
    """
    count = max(1, difficulty.target_count)
    if not candidates:
        return [_synthesize_fallback_point(difficulty, neglect_side) for _ in range(count)]

    neglected_count = round(count * NEGLECT_SIDE_BIAS)
    other_count = count - neglected_count

    def is_neglected_side(p: Vector3) -> bool:
        angle = _angle_deg(p)
        return (angle < 0) == (neglect_side == NeglectSide.left)

    neglected_pool = sorted(
        (p for p in candidates if is_neglected_side(p)),
        key=lambda p: _score_point(p, difficulty, neglect_side),
    )
    other_pool = sorted(
        (p for p in candidates if not is_neglected_side(p)),
        key=lambda p: _score_point(p, difficulty, neglect_side),
    )

    chosen = neglected_pool[:neglected_count] + other_pool[:other_count]

    # If either pool ran short (not enough real candidates on that side),
    # top up with synthesized points so the round always has the right count.
    while len(chosen) < count:
        chosen.append(_synthesize_fallback_point(difficulty, neglect_side))

    return chosen[:count]


def _synthesize_fallback_point(difficulty: DifficultyParams, neglect_side: NeglectSide) -> Vector3:
    """
    Used only if Unity couldn't supply any raycast-detected candidates this
    round (e.g. patient looking at a blank wall/floor with nothing tracked
    yet). Unity should treat this as a suggestion and re-validate against
    its own AR planes before actually spawning, never spawn blindly at an
    untracked point.
    """
    angle_deg = difficulty.eccentricity_deg if neglect_side == NeglectSide.right else -difficulty.eccentricity_deg
    angle_rad = math.radians(angle_deg)
    x = difficulty.distance_m * math.sin(angle_rad)
    z = difficulty.distance_m * math.cos(angle_rad)
    return Vector3(x=round(x, 3), y=0.0, z=round(z, 3))
