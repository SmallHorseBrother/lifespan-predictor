from dataclasses import replace

import numpy as np
import pytest

from lifespan_predictor import Profile, simulate
from lifespan_predictor.exposures import validate
from lifespan_predictor.scenarios import build_scenarios

TYPICAL_MAN = Profile(
    sex="male",
    age=30,
    height_cm=175,
    weight_kg=82,
    smoking="current_lt20",
    alcohol="daily_1_29",
    pa_met_h=14,
    fruit_g=60,
    fish_g=15,
    sleep_h=6.5,
)
TYPICAL_WOMAN = Profile(sex="female", age=40, height_cm=160, weight_kg=58, pa_met_h=12, fruit_g=90, fish_g=15)


@pytest.fixture(scope="module")
def result():
    return simulate(TYPICAL_MAN, n_lives=6000, n_param_draws=60, seed=11)


def _mean_remaining(res, name):
    return (res.results[name].death_age - res.profile.age).mean()


def test_scenarios_are_ordered(result):
    assert _mean_remaining(result, "optimized") > _mean_remaining(result, "current") > _mean_remaining(
        result, "adverse"
    )


def test_common_random_numbers_pair_lives(result):
    # Each life sees the same threshold in every future, so a better future
    # can only make a given life longer, never shorter.
    better = result.results["optimized"].death_age - result.results["current"].death_age
    worse = result.results["adverse"].death_age - result.results["current"].death_age
    assert (better >= -1e-9).mean() > 0.999
    assert (worse <= 1e-9).mean() > 0.999


def test_same_seed_is_reproducible():
    a = simulate(TYPICAL_WOMAN, n_lives=2000, n_param_draws=20, seed=5)
    b = simulate(TYPICAL_WOMAN, n_lives=2000, n_param_draws=20, seed=5)
    assert np.array_equal(a.results["current"].death_age, b.results["current"].death_age)


def test_different_seed_gives_different_lives():
    a = simulate(TYPICAL_WOMAN, n_lives=2000, n_param_draws=20, seed=5)
    b = simulate(TYPICAL_WOMAN, n_lives=2000, n_param_draws=20, seed=6)
    assert not np.array_equal(a.results["current"].death_age, b.results["current"].death_age)


def test_monte_carlo_mean_matches_analytic_expectation(result):
    for r in result.results.values():
        remaining = r.death_age - result.profile.age
        se = remaining.std() / np.sqrt(remaining.size)
        assert remaining.mean() == pytest.approx(r.expected_remaining_by_draw.mean(), abs=4 * se + 0.05)


def test_survival_curves_and_weights(result):
    for name, r in result.results.items():
        assert r.survival[0] == pytest.approx(1.0)
        assert np.all(np.diff(r.survival) <= 0)
        assert r.weight_kg.shape == r.ages.shape
    assert result.results["current"].weight_kg[-1] == pytest.approx(82, abs=1e-6)
    assert result.results["adverse"].weight_kg[-1] > 85  # +250 kcal/d surplus
    bmi_end = result.results["optimized"].bmi[-1]
    assert 20.5 - 0.05 <= bmi_end <= 23.5 + 0.05


def test_lag_delays_benefit():
    quick = simulate(TYPICAL_MAN, n_lives=3000, n_param_draws=30, seed=3, lag_tau_years=0)
    slow = simulate(TYPICAL_MAN, n_lives=3000, n_param_draws=30, seed=3, lag_tau_years=10)
    gain = lambda r: r.paired_difference("optimized")["mean_years"]  # noqa: E731
    assert gain(quick) > gain(slow) > 0


def test_point_estimate_mode_has_no_model_spread():
    r = simulate(TYPICAL_WOMAN, n_lives=1000, parameter_uncertainty=False, seed=2)
    spread = r.results["current"].expected_remaining_by_draw
    assert spread.size == 1


def test_summary_is_json_ready_and_carries_disclaimer(result):
    import json

    payload = result.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["disclaimer"] == "模拟估算，不用于诊断"
    assert {s["scenario"] for s in payload["scenarios"]} == {"current", "optimized", "adverse"}
    assert payload["paired_vs_reference"]["optimized"]["mean_years"] > 0


def test_optimized_scenario_flags_extrapolated_fruit(result):
    flags = " ".join(result.results["optimized"].flags)
    assert "水果" in flags


def test_scenarios_never_make_bad_habits_better():
    heavy = replace(TYPICAL_MAN, smoking="current_ge20", alcohol="daily_ge60", pa_met_h=2, fruit_g=0, sleep_h=4)
    _, opt, adv = build_scenarios(heavy)
    assert adv.pa_met_h == 2 and adv.fruit_g == 0 and adv.sleep_h == 4
    assert opt.smoking == "former" and opt.alcohol == "less_than_daily"


@pytest.mark.parametrize(
    "change",
    [{"sex": "other"}, {"age": 5}, {"weight_kg": 10}, {"smoking": "sometimes"}, {"sleep_h": -1}],
)
def test_invalid_inputs_are_rejected(change):
    with pytest.raises(ValueError):
        validate(replace(TYPICAL_MAN, **change))


def test_remaining_life_is_continuous_across_a_birthday():
    def mean_remaining(age):
        r = simulate(replace(TYPICAL_WOMAN, age=age), n_lives=4000, parameter_uncertainty=False, seed=9)
        return r.results["current"].expected_remaining_by_draw[0]

    ages = np.array([40.0, 40.25, 40.49, 40.5, 40.75, 40.99, 41.0])
    remaining = np.array([mean_remaining(a) for a in ages])
    steps = np.diff(remaining) / np.diff(ages)
    # remaining life falls by a bit under one year per year of age, smoothly
    assert np.all((steps < -0.8) & (steps > -1.0))


def test_hazard_path_uses_attained_age_and_recentering():
    # Independent recomputation of year-k hazard for a non-integer start age:
    # m(a) * exp(att(a) * lnHR) / mean_RR(a) at the interval midpoint age.
    from lifespan_predictor.hazards import (
        age_attenuation,
        alcohol_code,
        behaviour_log_hr,
        bmi_log_hr,
        point_coefficients,
        smoking_code,
    )
    from lifespan_predictor.life_table import baseline_mx
    from lifespan_predictor.population import mean_rr_by_age, reference_population
    from lifespan_predictor.scenarios import current
    from lifespan_predictor.simulate import scenario_hazard

    profile = replace(TYPICAL_MAN, age=62.7)
    c = point_coefficients("male")
    mean_rr = mean_rr_by_age(reference_population("male"), c)
    scn = current(profile)
    hazard, _, _ = scenario_hazard(profile, scn, scn, c, mean_rr, lag_tau=0)
    lhr = behaviour_log_hr(
        np.array([smoking_code(scn.smoking)]),
        np.array([alcohol_code(scn.alcohol)]),
        np.array([scn.pa_met_h]),
        np.array([scn.fruit_g]),
        np.array([scn.fish_g]),
        np.array([scn.sleep_h]),
        c,
    )[0, 0] + bmi_log_hr(np.array([profile.bmi]), c)[0, 0]
    for k in (0, 20):
        mid = profile.age + k + 0.5  # 63.2 -> between knots 62.5 (m_62) and 63.5 (m_63)
        lo = int(np.floor(mid - 0.5))
        w = mid - 0.5 - lo
        mx = baseline_mx("male")
        base = mx[lo] ** (1 - w) * mx[lo + 1] ** w
        rr = mean_rr[0, lo] ** (1 - w) * mean_rr[0, lo + 1] ** w
        expected = base * np.exp(age_attenuation("male", np.array([mid]))[0] * lhr) / rr
        assert hazard[0, k] == pytest.approx(expected, rel=1e-9)


def test_lives_must_split_evenly_over_parameter_draws():
    with pytest.raises(ValueError):
        simulate(TYPICAL_WOMAN, n_lives=150, n_param_draws=100)
    r = simulate(TYPICAL_WOMAN, n_lives=40, n_param_draws=100, seed=1)
    assert r.n_param_draws == 40 and r.results["current"].death_age.size == 40
