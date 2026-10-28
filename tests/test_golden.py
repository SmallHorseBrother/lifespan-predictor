"""Golden fixtures and exported constants stay in sync with the engine."""

import json

import pytest

from lifespan_predictor.exposures import ALCOHOL, SMOKING, Profile
from lifespan_predictor.foodlink import ENGINE_VERSION, one_life
from scripts import export_constants, export_golden

FIXTURES = json.loads(export_golden.OUT.read_text(encoding="utf-8"))
CASES = FIXTURES["cases"]


def test_enough_cases_and_coverage():
    assert len(CASES) >= 200
    assert FIXTURES["engine_version"] == ENGINE_VERSION
    profiles = [c["profile"] for c in CASES]
    assert {round(p["age"]) for p in profiles} >= {18, 30, 50, 79, 85}
    assert {p["sex"] for p in profiles} == {"male", "female"}
    assert {p["smoking"] for p in profiles} == set(SMOKING)
    assert {p["alcohol"] for p in profiles} == set(ALCOHOL)
    bmis = [p["weight_kg"] / (p["height_cm"] / 100) ** 2 for p in profiles]
    assert min(bmis) <= 16.1 and max(bmis) >= 39.9
    assert any(p["pa_met_h"] > 59.3 or p["fruit_g"] > 300 or p["sleep_h"] < 4 for p in profiles)
    starts = {c["change_start_age"] for c in CASES}
    assert None in starts and 99.0 in starts and any(s and 40 <= s <= 95 for s in starts)
    kinds = {json.dumps(c["adherence"], sort_keys=True) for c in CASES}
    assert {"null", "0.0", "0.5"} <= kinds and any("decay" in k for k in kinds)
    assert {c["mode"] for c in CASES} == {"diet", "lifestyle"}
    assert any(c["custom_scenario"] for c in CASES)


@pytest.mark.parametrize("case", CASES[::8], ids=lambda c: c["id"])
def test_fixture_reproduces(case):
    fresh = export_golden._rounded(export_golden.case_outputs(case))
    tol = FIXTURES["tolerance"]
    for key, expected in case["expected"].items():
        got = fresh[key]
        for q, v in expected["life_expectancy"].items():
            assert got["life_expectancy"][q] == pytest.approx(v, abs=tol["age_years"] / 50)
        for age, v in expected["p_alive"].items():
            assert (got["p_alive"][age] is None) == (v is None)
            if v is not None:
                assert got["p_alive"][age] == pytest.approx(v, abs=tol["p_alive"] / 50)
        assert got["one_life"] == pytest.approx(expected["one_life"], abs=tol["age_years"] / 50)


@pytest.mark.parametrize("case", [c for c in CASES if not c["custom_scenario"]][:3], ids=lambda c: c["id"])
def test_one_life_api_matches_fixture(case):
    profile = Profile(**case["profile"])
    for j in range(3):
        life = one_life(
            profile, mode=case["mode"], seed=case["seed"], life_index=j,
            change_start_age=case["change_start_age"], adherence=case["adherence"],
        )
        for branch in life["branches"]:
            assert branch["death_age"] == pytest.approx(case["expected"][branch["key"]]["one_life"][j], abs=0.006)


def test_constants_file_is_up_to_date():
    assert export_constants.OUT.read_text(encoding="utf-8") == export_constants.render(), "run scripts/export_constants.py"
