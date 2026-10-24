import numpy as np
import pytest

from lifespan_predictor.life_table import MAX_AGE, baseline_mx, life_expectancy, survival_curve
from lifespan_predictor.registry import load_registry


@pytest.mark.parametrize("sex", ["male", "female"])
def test_calibrated_e0_matches_official_2020(sex):
    official = load_registry().value("baseline.official_e0_2020", sex)
    assert life_expectancy(baseline_mx(sex)) == pytest.approx(official, abs=0.01)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_adult_mortality_rises_with_age(sex):
    mx = baseline_mx(sex)
    assert mx.shape == (MAX_AGE + 1,)
    assert np.all(np.diff(mx[40:]) > 0)


def test_women_outlive_men():
    assert life_expectancy(baseline_mx("female"), 30) > life_expectancy(baseline_mx("male"), 30)


def test_baseline_table_is_read_only():
    with pytest.raises(ValueError):
        baseline_mx("male")[50] = 0.0


def test_survival_curve_is_monotone_from_one():
    s = survival_curve(baseline_mx("male"), 30)
    assert s[0] == 1.0
    assert np.all(np.diff(s) <= 0)
