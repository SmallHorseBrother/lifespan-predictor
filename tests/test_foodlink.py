"""FoodLink contract: field names, diet-only mode, determinism, portable RNG."""

import json
from dataclasses import replace

import pytest

from lifespan_predictor import Profile, simulate
from lifespan_predictor.foodlink import life_uniform, lifespan_result, one_life, splitmix64
from lifespan_predictor.scenarios import build_scenarios

PERSON = Profile(
    sex="male", age=32.4, height_cm=175, weight_kg=78.2, alcohol="daily_1_29", pa_met_h=14, fruit_g=40, fish_g=10, sleep_h=6.3
)
NON_DIET = ("smoking", "alcohol", "pa_met_h", "sleep_h")


@pytest.fixture(scope="module")
def result():
    return lifespan_result(PERSON, mode="diet", n_param_draws=60, seed=7)


def test_diet_mode_only_changes_diet():
    current, right, loose = build_scenarios(replace(PERSON, smoking="current_lt20"), "diet")
    assert (current.name, right.name, loose.name) == ("current", "diet_right", "diet_loose")
    for field in NON_DIET:
        assert getattr(current, field) == getattr(right, field) == getattr(loose, field)
    assert right.fruit_g > current.fruit_g > loose.fruit_g
    assert loose.delta_intake_kcal > 0 > right.delta_intake_kcal  # BMI 25.5 -> back into range


def test_lifestyle_mode_still_available():
    keys = [s.name for s in build_scenarios(PERSON, "lifestyle")]
    assert keys == ["current", "optimized", "adverse"]
    with pytest.raises(ValueError):
        build_scenarios(PERSON, "everything")


def test_result_has_contract_fields(result):
    json.dumps(result, ensure_ascii=False)
    assert set(result["model"]) >= {"engine_version", "params_version", "params_sha256", "input_hash", "seed", "computed_at"}
    assert set(result["applicability"]) == {"age_in_range", "extrapolated_inputs", "completeness_pct"}
    assert [b["key"] for b in result["branches"]] == ["current", "diet_right", "diet_loose"]
    for b in result["branches"]:
        assert set(b["life_expectancy"]) == {"median", "mean", "p10", "p90"}
        assert set(b["survival"][0]) == {"age", "p_alive", "p_alive_disease_free"}
        assert set(b["trajectory"][0]) == {"age", "weight_kg", "bmi"}
        assert set(b["death_age_histogram"]) >= {"bin_years", "start_age", "density"}
        assert sum(b["death_age_histogram"]["density"]) == pytest.approx(1.0, abs=0.002)
    for change in result["top_changes"]:
        assert set(change) >= {
            "factor", "label", "current", "target", "unit", "gain_years", "gain_p85_pct", "evidence", "parameter_ids"
        }
        assert change["gain_years"]["low"] <= change["gain_years"]["median"] <= change["gain_years"]["high"]
    assert all({"code", "field", "message"} <= set(w) for w in result["warnings"])


def test_branch_order_and_quantiles(result):
    mean = {b["key"]: b["life_expectancy"]["mean"] for b in result["branches"]}
    assert mean["diet_right"] > mean["current"] > mean["diet_loose"]
    for b in result["branches"]:
        le = b["life_expectancy"]
        assert le["p10"] < le["median"] < le["p90"]


def test_deterministic_curves_match_monte_carlo():
    det = lifespan_result(PERSON, mode="diet", n_param_draws=20)["branches"][0]["life_expectancy"]["mean"]
    mc = simulate(PERSON, n_lives=20000, n_param_draws=1, parameter_uncertainty=False, mode="diet", seed=3)
    assert mc.results["current"].death_age.mean() == pytest.approx(det, abs=0.25)


def test_input_hash_tracks_inputs_not_seed(result):
    same = lifespan_result(PERSON, mode="diet", n_param_draws=60, seed=8)
    other = lifespan_result(replace(PERSON, fruit_g=41), mode="diet", n_param_draws=60, seed=7)
    assert same["model"]["input_hash"] == result["model"]["input_hash"]
    assert other["model"]["input_hash"] != result["model"]["input_hash"]
    assert len(result["model"]["params_sha256"]) == 64


def test_splitmix64_reference_values():
    # Reference SplitMix64 outputs for states 0 and 1; the Go port must match.
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(1) == 0x910A2DEC89025CC1
    u = [life_uniform(20261024, i) for i in range(1000)]
    assert 0 < min(u) and max(u) < 1
    assert sum(u) / len(u) == pytest.approx(0.5, abs=0.03)


def test_one_life_shares_random_numbers_across_branches():
    life = one_life(PERSON, mode="diet", seed=11, life_index=3)
    ages = {b["key"]: b["death_age"] for b in life["branches"]}
    assert ages["diet_right"] >= ages["current"] >= ages["diet_loose"]
    assert all(b["events"][-1] == {"age": b["death_age"], "type": "death"} for b in life["branches"])
    assert one_life(PERSON, mode="diet", seed=11, life_index=3) == life
