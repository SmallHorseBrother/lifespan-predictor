"""Calibration anchor checks (the release gate for model changes).

Acceptance bands:
- Nat Commun 2025 combined ideal / practical scenarios: inside the paper's
  own 95% CI (both sexes).
- Lancet Public Health 2022 five vs 0-1 low-risk factors: men inside the
  paper's 95% CI; both sexes within 0.6x-1.5x of the point estimate. Women
  currently run about +34% (see README "校准锚点").
"""

import pytest

from lifespan_predictor.anchors import anchor_report
from lifespan_predictor.registry import load_registry


@pytest.fixture(scope="module")
def report():
    return anchor_report()


@pytest.mark.parametrize("sex", ["male", "female"])
@pytest.mark.parametrize("scenario", ["ideal", "practical"])
def test_nc2025_combined_scenarios_inside_paper_ci(report, sex, scenario):
    lo, hi = load_registry()["anchor.nc2025"].range[sex][scenario]
    gain = report["nc2025"][sex][scenario]["model"]
    assert lo <= gain <= hi, f"{sex} {scenario}: model {gain:.2f} outside paper CI [{lo}, {hi}]"


@pytest.mark.parametrize("sex", ["male", "female"])
def test_nc2025_single_factors_same_direction_and_order_of_magnitude(report, sex):
    rows = report["nc2025"][sex]
    for name in ("ideal_smoking", "ideal_pa", "ideal_diet"):
        model, paper = rows[name]["model"], rows[name]["paper"]
        assert model > 0
        assert paper / 3 <= model <= paper * 3, f"{sex} {name}: {model:.2f} vs {paper}"
    # diet is the largest single lever in both the paper and the model
    assert rows["ideal_diet"]["model"] == max(rows[n]["model"] for n in ("ideal_smoking", "ideal_pa", "ideal_diet"))


def test_ckb2022_men_inside_paper_ci(report):
    lo, hi = load_registry()["anchor.ckb2022"].range["male"]["le_gain"]
    assert lo <= report["ckb2022"]["male"]["gain"] <= hi
    hr_lo, hr_hi = load_registry()["anchor.ckb2022"].range["male"]["hr_5_vs_0_1"]
    assert hr_lo <= report["ckb2022"]["male"]["rr_all5_vs_0_1"] <= hr_hi


@pytest.mark.parametrize("sex", ["male", "female"])
def test_ckb2022_total_effect_order_of_magnitude(report, sex):
    row = report["ckb2022"][sex]
    assert 0.6 * row["paper_gain"] <= row["gain"] <= 1.5 * row["paper_gain"]


@pytest.mark.parametrize("sex", ["male", "female"])
def test_baseline_matches_official_life_expectancy(report, sex):
    row = report["baseline"][sex]
    assert row["e0"] == pytest.approx(row["e0_official"], abs=0.01)
