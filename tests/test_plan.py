"""Delayed start (change_start_age) and adherence."""

from dataclasses import replace

import pytest

from lifespan_predictor import Profile
from lifespan_predictor.foodlink import gain_by_start_age, input_hash, lifespan_result, one_life
from lifespan_predictor.plan import Adherence, ChangePlan, make_plan
from lifespan_predictor.weight import simulate_weight

PERSON = Profile(
    sex="male", age=30, height_cm=175, weight_kg=82, smoking="current_lt20", alcohol="daily_1_29",
    pa_met_h=14, fruit_g=60, fish_g=15, sleep_h=6.5,
)
FADE = {"type": "decay", "start": 0.9, "end": 0.5, "half_life_years": 2}


def means(result) -> dict:
    return {b["key"]: b["life_expectancy"]["mean"] for b in result["branches"]}


def weights_end(result) -> dict:
    return {b["key"]: b["trajectory"][-1]["weight_kg"] for b in result["branches"]}


@pytest.fixture(scope="module")
def default():
    return lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, seed=1)


def test_explicit_defaults_equal_plain_call(default):
    explicit = lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, seed=1, change_start_age=30, adherence=1.0)
    assert explicit["branches"] == default["branches"]
    assert explicit["top_changes"] == default["top_changes"]
    assert explicit["model"]["input_hash"] == default["model"]["input_hash"]


def test_zero_adherence_or_start_at_110_equals_current(default):
    for kwargs in ({"adherence": 0.0}, {"change_start_age": 110}):
        r = lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, **kwargs)
        m = means(r)
        assert m["optimized"] == pytest.approx(m["current"], abs=1e-9)
        assert m["adverse"] == pytest.approx(m["current"], abs=1e-9)


def test_partial_adherence_sits_between(default):
    half = means(lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, adherence=0.5))
    full = means(default)
    assert full["current"] < half["optimized"] < full["optimized"]
    assert full["adverse"] < half["adverse"] < full["current"]
    fade = means(lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, adherence=FADE))
    assert half["optimized"] < fade["optimized"] < means(lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, adherence=0.9))["optimized"]


def test_later_start_gains_less_but_still_gains():
    rows = gain_by_start_age(PERSON, mode="diet", ages=[30, 40, 50, 60, 70, 80])["rows"]
    gains = [r["gain_years"] for r in rows]
    assert gains == sorted(gains, reverse=True)
    assert gains[-1] > 0
    assert rows[0]["gain_years"] == pytest.approx(
        means(lifespan_result(PERSON, mode="diet", n_param_draws=20))["diet_right"]
        - means(lifespan_result(PERSON, mode="diet", n_param_draws=20))["current"],
        abs=0.011,
    )


def test_gain_by_start_age_defaults_and_validation():
    out = gain_by_start_age(replace(PERSON, age=45))
    assert [r["change_start_age"] for r in out["rows"]] == [45.0, 50.0, 60.0, 70.0]
    assert out["branch"] == "diet_right" and out["reference"] == "current"
    with pytest.raises(ValueError):
        gain_by_start_age(PERSON, branch="nope")


def test_weight_holds_until_start_then_follows_new_balance():
    late = lifespan_result(PERSON, mode="diet", n_param_draws=20, change_start_age=50)
    right = next(b for b in late["branches"] if b["key"] == "diet_right")
    by_age = {round(p["age"]): p["weight_kg"] for p in right["trajectory"]}
    assert by_age[49] == pytest.approx(82.0, abs=1e-9)
    assert by_age[55] < 80
    assert weights_end(late)["diet_right"] == pytest.approx(weights_end(lifespan_result(PERSON, mode="diet", n_param_draws=20))["diet_right"], abs=0.05)


def test_weight_schedule_constant_matches_plain_constant():
    plain = simulate_weight("female", 40, 160, 70, -150, 10)
    scheduled = simulate_weight("female", 40, 160, 70, -150, 10, delta_schedule=lambda t: -150.0)
    assert list(plain.weight_kg) == list(scheduled.weight_kg)


def test_policy_and_hash_echo_the_plan(default):
    r = lifespan_result(PERSON, mode="lifestyle", n_param_draws=20, change_start_age=45, adherence=FADE)
    assert r["plan"] == {"change_start_age": 45.0, "adherence": {**FADE, "half_life_years": 2.0, "start": 0.9, "end": 0.5}}
    assert all(b["policy"]["change_start_age"] == 45.0 for b in r["branches"])
    assert default["branches"][0]["policy"]["adherence"] == {"type": "constant", "level": 1.0}
    assert r["model"]["input_hash"] != default["model"]["input_hash"]
    assert input_hash(PERSON, "lifestyle", 2.0, 20) == input_hash(PERSON, "lifestyle", 2.0, 20, ChangePlan())


def test_one_life_shares_threshold_under_a_plan():
    life = one_life(PERSON, mode="lifestyle", seed=3, life_index=2, change_start_age=50, adherence=0.7)
    ages = {b["key"]: b["death_age"] for b in life["branches"]}
    assert ages["optimized"] >= ages["current"] >= ages["adverse"]
    assert life["plan"]["change_start_age"] == 50.0
    early_death = min(ages.values()) < 50
    if early_death:  # a life that ends before the change starts is the same in every branch
        assert len(set(ages.values())) == 1


def test_adherence_specs_and_validation():
    assert Adherence.from_spec(0.7).to_dict() == {"type": "constant", "level": 0.7}
    fade = Adherence.from_spec(FADE)
    assert fade.level(0.0) == pytest.approx(0.9) and fade.level(2.0) == pytest.approx(0.7) and fade.level(200.0) == pytest.approx(0.5)
    for bad in (1.5, -0.1, {"type": "decay", "start": 0.9, "end": 0.5, "half_life_years": 0}, {"type": "weird"}):
        with pytest.raises(ValueError):
            Adherence.from_spec(bad)
    with pytest.raises(ValueError):
        make_plan(change_start_age=20).validate(30)
    with pytest.raises(ValueError):
        lifespan_result(PERSON, change_start_age=120)
