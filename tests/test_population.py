import numpy as np
import pytest

from lifespan_predictor.hazards import point_coefficients
from lifespan_predictor.population import BUCKET_STARTS, mean_rr_by_age, reference_population
from lifespan_predictor.registry import load_registry


@pytest.mark.parametrize("sex", ["male", "female"])
def test_reference_population_reproduces_published_cnhs_aggregates(sex):
    # The synthetic population must match the published inputs it was built from.
    reg = load_registry()
    pop = reference_population(sex)
    current = np.isin(pop.smoking, (2, 3))
    by_age = reg.value("pop.cnhs_current_smoking_by_age", sex)
    young = pop.bucket <= 1  # 30-39
    assert current[young].mean() * 100 == pytest.approx(by_age[0], abs=1.5)
    pa_mean = reg.value("pop.cnhs_pa_mean_by_age", sex)[0]
    assert pop.pa_met_h[young].mean() == pytest.approx(pa_mean, rel=0.08)
    bmi_shares = reg.value("pop.cnhs_bmi", sex)
    assert (pop.bmi >= 28).mean() * 100 == pytest.approx(bmi_shares[3], abs=1.0)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_mean_rr_is_positive_and_smoking_heavier_populations_score_higher(sex):
    c = point_coefficients(sex)
    pop = reference_population(sex)
    base = mean_rr_by_age(pop, c)[0]
    assert np.all(base > 0)
    everyone_smokes = pop.with_changes(smoking=np.full_like(pop.smoking, 3))
    assert np.all(mean_rr_by_age(everyone_smokes, c)[0][30:] > base[30:])
    assert len(BUCKET_STARTS) == 12
