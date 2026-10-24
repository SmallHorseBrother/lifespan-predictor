"""Hall-type energy-balance model checked against Hall 2011 (Lancet)."""

import numpy as np
import pytest

from lifespan_predictor.weight import (
    KCAL_TO_MJ,
    intake_delta_for_target_weight,
    simulate_weight,
    steady_state_weight,
)


def test_hall_2011_example_100kg_man_cutting_2mj_plateaus_near_75kg():
    # Hall 2011: 100 kg sedentary man, -2 MJ/d (480 kcal) -> plateau ~75 kg,
    # half of the loss in ~1 year, 95% after ~3 years.
    traj = simulate_weight("male", 40, 180, 100, -480, n_years=10)
    assert traj.weight_kg[-1] == pytest.approx(75, abs=2.5)
    total = traj.weight_kg[0] - traj.weight_kg[-1]
    half_year = np.interp(total / 2, traj.weight_kg[0] - traj.weight_kg, traj.years)
    assert 0.5 <= half_year <= 1.5
    assert (traj.weight_kg[0] - traj.weight_kg[3]) / total >= 0.9


@pytest.mark.parametrize("sex,weight,height", [("male", 70, 170), ("male", 85, 170), ("female", 65, 160)])
def test_rule_of_thumb_about_1kg_per_100kj_per_day(sex, weight, height):
    kcal_100kj = 0.1 / KCAL_TO_MJ
    change = steady_state_weight(sex, 40, height, weight, kcal_100kj) - weight
    assert 0.8 <= change <= 1.6


def test_simulation_converges_to_analytic_steady_state():
    traj = simulate_weight("female", 35, 162, 60, 250, n_years=15)
    assert traj.weight_kg[-1] == pytest.approx(steady_state_weight("female", 35, 162, 60, 250), abs=0.1)


def test_zero_imbalance_keeps_weight_constant():
    traj = simulate_weight("male", 30, 175, 82, 0.0, n_years=5)
    assert np.allclose(traj.weight_kg, 82.0)


def test_intake_for_target_weight_round_trips():
    delta = intake_delta_for_target_weight("male", 45, 172, 90, 70)
    assert delta < 0
    assert steady_state_weight("male", 45, 172, 90, delta) == pytest.approx(70, abs=0.05)


def test_heavier_people_lose_more_for_same_deficit():
    light = steady_state_weight("male", 40, 180, 80, -480)
    heavy = steady_state_weight("male", 40, 180, 100, -480)
    assert (100 - heavy) > (80 - light)
