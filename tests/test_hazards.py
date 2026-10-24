import numpy as np
import pytest

from lifespan_predictor.hazards import (
    age_attenuation,
    alcohol_log_hr,
    bmi_log_hr,
    diet_log_hr,
    draw_coefficients,
    pa_log_hr,
    point_coefficients,
    sleep_log_hr,
    smoking_log_hr,
)


@pytest.fixture(params=["male", "female"])
def coefs(request):
    return point_coefficients(request.param)


def test_smoking_risk_ordered_never_former_light_heavy(coefs):
    lhr = smoking_log_hr(np.array([0, 1, 2, 3]), coefs)[0]
    assert lhr[0] == 0 and np.all(np.diff(lhr) > 0)


def test_heavy_drinking_is_worst(coefs):
    lhr = alcohol_log_hr(np.array([0, 1, 2, 3]), coefs)[0]
    assert lhr[0] == 0 and lhr[3] == lhr.max()


def test_more_activity_lowers_risk_and_floor_applies(coefs):
    lhr = pa_log_hr(np.array([0.0, 2.0, 10.0, 20.0, 40.0]), coefs)[0]
    assert lhr[0] == lhr[1]  # floor 2 MET-h/d
    assert np.all(np.diff(lhr[1:]) < 0)
    assert lhr[3] == pytest.approx(0.0)


def test_diet_effect_is_capped(coefs):
    capped = diet_log_hr(np.array([coefs.fruit_cap, 2000.0]), np.array([20.0, 20.0]), coefs)[0]
    assert capped[0] == pytest.approx(capped[1])


def test_bmi_curve_lowest_between_20_and_25(coefs):
    bmis = np.array([16.0, 19.0, 22.0, 24.9, 30.0, 35.0])
    lhr = bmi_log_hr(bmis, coefs)[0]
    assert lhr[2] == lhr[3] == 0.0
    assert lhr[0] > lhr[1] > 0 and lhr[5] > lhr[4] > 0
    assert np.exp(lhr[4]) == pytest.approx(1.39, abs=1e-4)  # GBMC east Asia per 5 units


def test_sleep_reference_is_7_to_8_hours_and_shrunk(coefs):
    lhr = sleep_log_hr(np.array([5.0, 6.5, 7.5, 8.5, 9.5]), coefs)[0]
    assert lhr[2] == 0.0
    assert lhr[0] == pytest.approx(0.5 * np.log(1.12), rel=1e-3)
    assert lhr[4] == pytest.approx(0.5 * np.log(1.30), rel=1e-3)
    assert 0 < lhr[1] < lhr[0] and 0 < lhr[3] < lhr[4]


def test_age_attenuation_shape():
    att = age_attenuation("male", np.array([30, 80, 90, 100, 110]))
    assert list(att) == pytest.approx([1.0, 1.0, 0.75, 0.5, 0.5])


def test_parameter_draws_center_on_point_estimates():
    rng = np.random.default_rng(1)
    draws = draw_coefficients("male", 4000, rng)
    point = point_coefficients("male")
    assert draws.smk_ge20.mean() == pytest.approx(point.smk_ge20[0, 0], abs=0.003)
    assert draws.pa_ln.std() > 0
