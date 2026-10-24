import json
import math

import pytest

from lifespan_predictor.registry import REGISTRY_PATH, STATUSES, load_registry
from scripts.export_registry_csv import CSV_PATH, render_csv

REQUIRED_FIELDS = (
    "id",
    "domain",
    "variable",
    "exposure_definition",
    "reference_level",
    "unit",
    "effect_type",
    "estimate",
    "range",
    "population",
    "source",
    "evidence_type",
    "status",
    "estimated",
    "estimate_note",
)


@pytest.fixture(scope="module")
def raw():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_every_parameter_has_required_fields(raw):
    for p in raw["parameters"]:
        missing = [f for f in REQUIRED_FIELDS if f not in p]
        assert not missing, f"{p.get('id')} missing {missing}"


def test_status_is_paper_or_estimate_and_estimates_explain_themselves(raw):
    for p in raw["parameters"]:
        assert p["status"] in STATUSES
        assert p["estimated"] == (p["status"] == "估算")
        if p["status"] == "估算":
            assert p["estimate_note"] == p["rationale"]
            assert p["source"] == "estimate"
            assert len(p.get("rationale", "")) > 20, p["id"]
        else:
            assert p["source"] != "estimate", p["id"]


def test_paper_sources_carry_an_identifier_and_verification(raw):
    for key, src in raw["sources"].items():
        if key == "estimate":
            continue
        assert src.get("doi") or src.get("url"), key
        assert src.get("verified", "").startswith("2026-"), key


def test_log_hr_point_estimates_lie_inside_their_ranges():
    reg = load_registry()
    for p in reg.parameters.values():
        if p.effect_type not in ("log_hr", "log_hr_per_unit", "log_hr_per_ln_unit") or p.range is None:
            continue
        for sex in ("male", "female"):
            lo, hi = p.range[sex]
            assert lo <= p.value(sex) <= hi, f"{p.id} {sex}: {p.value(sex)} not in [{lo}, {hi}]"


def test_diet_unit_interpretation_matches_table1_hr_per_10g():
    # Sun 2025 Table 1: fruit HR 0.98 (both), fish 0.96 men / 0.93 women per 10 g/d.
    reg = load_registry()
    for pid, sex, table1 in (
        ("diet.fruit_per_g", "male", 0.98),
        ("diet.fruit_per_g", "female", 0.98),
        ("diet.fish_per_g", "male", 0.96),
        ("diet.fish_per_g", "female", 0.93),
    ):
        assert math.exp(10 * reg.value(pid, sex)) == pytest.approx(table1, abs=0.011)


def test_csv_export_is_up_to_date():
    assert CSV_PATH.read_text(encoding="utf-8-sig") == render_csv(), "run scripts/export_registry_csv.py"
