"""Log hazard ratios for the six lifestyle domains.

Effect structure (total-effect mode, no double counting by design):
- smoking, alcohol, physical activity, fruit, fish: CKB 2025 mortality model
  (Sun 2025 Nat Commun). These coefficients are adjusted for BMI and blood
  pressure, i.e. they exclude the path through body weight.
- BMI: GBMC 2016 (never-smokers, total effect of adiposity). Body weight is
  driven by the Hall-type energy model, so the weight path is counted once.
- sleep: Cappuccio 2010 meta-analysis, shrunk (registry sleep.shrinkage).

All functions broadcast: coefficient arrays have shape (D, 1) for D parameter
draws, exposure arrays have shape (N,) or (D, N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exposures import ALCOHOL, SMOKING
from .registry import load_registry

_DRAWN = {
    "smk_former": "smoking.former",
    "smk_lt20": "smoking.current_lt20",
    "smk_ge20": "smoking.current_ge20",
    "alc_1_29": "alcohol.daily_1_29",
    "alc_30_59": "alcohol.daily_30_59",
    "alc_ge60": "alcohol.daily_ge60",
    "pa_ln": "pa.ln_met_h",
    "fruit": "diet.fruit_per_g",
    "fish": "diet.fish_per_g",
    "bmi_low": "bmi.15_to_18_5",
    "bmi_185": "bmi.18_5_to_20",
    "bmi_per5": "bmi.per5_above25_east_asia",
    "sleep_short": "sleep.short",
    "sleep_long": "sleep.long",
}


@dataclass(frozen=True)
class Coefficients:
    """Coefficient arrays of shape (D, 1); D = 1 for point estimates."""

    smk_former: np.ndarray
    smk_lt20: np.ndarray
    smk_ge20: np.ndarray
    alc_1_29: np.ndarray
    alc_30_59: np.ndarray
    alc_ge60: np.ndarray
    pa_ln: np.ndarray
    fruit: np.ndarray
    fish: np.ndarray
    bmi_low: np.ndarray
    bmi_185: np.ndarray
    bmi_per5: np.ndarray
    sleep_short: np.ndarray
    sleep_long: np.ndarray
    # fixed structural values (not drawn)
    pa_center: float
    pa_floor: float
    fruit_center: float
    fish_center: float
    fruit_cap: float
    fish_cap: float
    sleep_cut: tuple[float, ...]
    sleep_shrink: float

    @property
    def n_draws(self) -> int:
        return self.pa_ln.shape[0]


def _fixed(sex: str) -> dict:
    reg = load_registry()
    centers = reg.value("diet.centers", sex)
    caps = reg.value("diet.effect_caps", sex)
    return {
        "pa_center": float(reg.value("pa.center", sex)),
        "pa_floor": float(reg.value("pa.floor", sex)),
        "fruit_center": float(centers["fruit"]),
        "fish_center": float(centers["fish"]),
        "fruit_cap": float(caps["fruit"]),
        "fish_cap": float(caps["fish"]),
        "sleep_cut": tuple(float(x) for x in reg.value("sleep.cutpoints", sex)),
        "sleep_shrink": float(reg.value("sleep.shrinkage", sex)),
    }


def point_coefficients(sex: str) -> Coefficients:
    reg = load_registry()
    drawn = {k: np.array([[float(reg.value(pid, sex))]]) for k, pid in _DRAWN.items()}
    return Coefficients(**drawn, **_fixed(sex))


def draw_coefficients(sex: str, n_draws: int, rng: np.random.Generator) -> Coefficients:
    """Sample published coefficients from Normal(estimate, SE from 95% CI).

    Only 论文 coefficients are drawn; 估算 structural values stay fixed and are
    explored by sensitivity analysis instead.
    """
    reg = load_registry()
    drawn = {}
    for key, pid in _DRAWN.items():
        param = reg[pid]
        mean = float(param.value(sex))
        se = param.standard_error(sex)
        drawn[key] = rng.normal(mean, se, size=(n_draws, 1))
    return Coefficients(**drawn, **_fixed(sex))


def smoking_log_hr(codes: np.ndarray, c: Coefficients) -> np.ndarray:
    """codes: int array (N,) indexing SMOKING; returns (D, N)."""
    table = np.concatenate([np.zeros_like(c.smk_former), c.smk_former, c.smk_lt20, c.smk_ge20], axis=1)
    return table[:, np.atleast_1d(codes)]


def alcohol_log_hr(codes: np.ndarray, c: Coefficients) -> np.ndarray:
    """codes: int array (N,) indexing ALCOHOL; returns (D, N)."""
    table = np.concatenate([np.zeros_like(c.alc_1_29), c.alc_1_29, c.alc_30_59, c.alc_ge60], axis=1)
    return table[:, np.atleast_1d(codes)]


def pa_log_hr(met_h: np.ndarray, c: Coefficients) -> np.ndarray:
    clipped = np.maximum(np.asarray(met_h, dtype=float), c.pa_floor)
    return c.pa_ln * np.log(clipped / c.pa_center)


def diet_log_hr(fruit_g: np.ndarray, fish_g: np.ndarray, c: Coefficients) -> np.ndarray:
    fruit = np.minimum(np.asarray(fruit_g, dtype=float), c.fruit_cap)
    fish = np.minimum(np.asarray(fish_g, dtype=float), c.fish_cap)
    return c.fruit * (fruit - c.fruit_center) + c.fish * (fish - c.fish_center)


def sleep_log_hr(hours: np.ndarray, c: Coefficients) -> np.ndarray:
    _, short_full, short_half, long_half, long_full, _ = c.sleep_cut
    h = np.asarray(hours, dtype=float)
    weight_short = np.where(h < short_full, 1.0, np.where(h < short_half, 0.5, 0.0))
    weight_long = np.where(h >= long_full, 1.0, np.where(h >= long_half, 0.5, 0.0))
    return c.sleep_shrink * (weight_short * c.sleep_short + weight_long * c.sleep_long)


def bmi_log_hr(bmi: np.ndarray, c: Coefficients) -> np.ndarray:
    b = np.asarray(bmi, dtype=float)
    low = np.where(b < 18.5, c.bmi_low, np.where(b < 20.0, c.bmi_185, 0.0))
    high = c.bmi_per5 * np.maximum(b - 25.0, 0.0) / 5.0
    return low + high


def behaviour_log_hr(
    smoking: np.ndarray,
    alcohol: np.ndarray,
    pa_met_h: np.ndarray,
    fruit_g: np.ndarray,
    fish_g: np.ndarray,
    sleep_h: np.ndarray,
    c: Coefficients,
) -> np.ndarray:
    """Sum of all non-BMI log hazard ratios, shape (D, N)."""
    return (
        smoking_log_hr(np.asarray(smoking), c)
        + alcohol_log_hr(np.asarray(alcohol), c)
        + pa_log_hr(pa_met_h, c)
        + diet_log_hr(fruit_g, fish_g, c)
        + sleep_log_hr(sleep_h, c)
    )


def age_attenuation(sex: str, ages: np.ndarray) -> np.ndarray:
    """Multiplier applied to every log HR at a given attained age."""
    rule = load_registry().value("age.effect_attenuation", sex)
    a = np.asarray(ages, dtype=float)
    frac = np.clip((a - rule["start_age"]) / (rule["end_age"] - rule["start_age"]), 0.0, 1.0)
    return 1.0 - frac * (1.0 - rule["end_multiplier"])


def smoking_code(name: str) -> int:
    return SMOKING.index(name)


def alcohol_code(name: str) -> int:
    return ALCOHOL.index(name)
