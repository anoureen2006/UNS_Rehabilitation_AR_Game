"""
Run with: pytest test_difficulty_controller.py -v
"""

import difficulty_controller as dc
from schemas import DifficultyParams


def test_initial_difficulty_values():
    d = dc.initial_difficulty()
    assert d.speed == 0.5
    assert d.eccentricity_deg == 20.0
    assert d.distance_m == 1.5


def test_no_change_below_streak_threshold():
    current = dc.initial_difficulty()
    result = dc.next_difficulty(current, consecutive_hits=1, consecutive_misses=0)
    assert result.speed == current.speed
    assert result.eccentricity_deg == current.eccentricity_deg
    assert result.distance_m == current.distance_m


def test_three_hit_streak_increases_difficulty():
    current = dc.initial_difficulty()
    result = dc.next_difficulty(current, consecutive_hits=3, consecutive_misses=0)
    assert result.speed > current.speed
    assert result.eccentricity_deg > current.eccentricity_deg
    assert result.distance_m > current.distance_m


def test_two_miss_streak_decreases_difficulty():
    current = dc.initial_difficulty()
    result = dc.next_difficulty(current, consecutive_hits=0, consecutive_misses=2)
    assert result.speed < current.speed
    assert result.eccentricity_deg < current.eccentricity_deg
    assert result.distance_m < current.distance_m


def test_speed_never_exceeds_max():
    current = DifficultyParams(speed=dc.SPEED_MAX, eccentricity_deg=20, distance_m=1.5)
    result = dc.next_difficulty(current, consecutive_hits=3, consecutive_misses=0)
    assert result.speed <= dc.SPEED_MAX


def test_speed_never_below_min():
    current = DifficultyParams(speed=dc.SPEED_MIN, eccentricity_deg=20, distance_m=1.5)
    result = dc.next_difficulty(current, consecutive_hits=0, consecutive_misses=2)
    assert result.speed >= dc.SPEED_MIN


def test_eccentricity_never_exceeds_max():
    current = DifficultyParams(speed=0.5, eccentricity_deg=dc.ECC_MAX, distance_m=1.5)
    result = dc.next_difficulty(current, consecutive_hits=3, consecutive_misses=0)
    assert result.eccentricity_deg <= dc.ECC_MAX


def test_eccentricity_never_below_min():
    current = DifficultyParams(speed=0.5, eccentricity_deg=dc.ECC_MIN, distance_m=1.5)
    result = dc.next_difficulty(current, consecutive_hits=0, consecutive_misses=2)
    assert result.eccentricity_deg >= dc.ECC_MIN


def test_repeated_hit_streaks_eventually_plateau_at_max():
    current = dc.initial_difficulty()
    for _ in range(200):  # far more than enough steps to hit the ceiling
        current = dc.next_difficulty(current, consecutive_hits=3, consecutive_misses=0)
    assert current.speed == dc.SPEED_MAX
    assert current.eccentricity_deg == dc.ECC_MAX
    assert current.distance_m == dc.DIST_MAX


def test_target_count_and_time_limit_pass_through_unchanged():
    current = DifficultyParams(speed=0.5, eccentricity_deg=20, distance_m=1.5, target_count=5, time_limit_s=45.0)
    result = dc.next_difficulty(current, consecutive_hits=3, consecutive_misses=0)
    assert result.target_count == 5
    assert result.time_limit_s == 45.0
