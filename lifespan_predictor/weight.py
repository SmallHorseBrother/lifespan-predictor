"""Dynamic energy-balance body-weight model (Hall group, two compartments).

Fat mass F and lean mass L change with the daily energy imbalance:

    dF/dt = (1 - p) * imbalance / rho_F
    dL/dt =      p  * imbalance / rho_L
    p = C / (C + F),  C = 10.4 * rho_L / rho_F        (Forbes's law)
    imbalance = intake - E(F, L),  E = 0.14 L + 0.05 F + K

K is solved so that the starting body is in energy balance; a scenario then
shifts intake by a constant delta. Parameters: Chow & Hall 2008 (PLoS Comput
Biol); initial body fat: Deurenberg 1991. Checked against the Hall 2011
Lancet rule of thumb in tests/test_weight.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import load_registry

KCAL_TO_MJ = 4.184e-3
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class BodyParams:
    rho_fat: float
    rho_lean: float
    forbes: float
    ee_lean: float
    ee_fat: float

    @property
    def c(self) -> float:
        return self.forbes * self.rho_lean / self.rho_fat


@dataclass(frozen=True)
class WeightTrajectory:
    years: np.ndarray  # 0, 1, ..., n_years
    weight_kg: np.ndarray
    fat_kg: np.ndarray
    lean_kg: np.ndarray
    delta_intake_kcal: float


def body_params(sex: str) -> BodyParams:
    reg = load_registry()
    ee = reg.value("hall.energy_expenditure", sex)
    return BodyParams(
        rho_fat=float(reg.value("hall.rho_fat", sex)),
        rho_lean=float(reg.value("hall.rho_lean", sex)),
        forbes=float(reg.value("hall.forbes_c", sex)),
        ee_lean=float(ee["lean"]),
        ee_fat=float(ee["fat"]),
    )


def initial_fat_kg(sex: str, age: float, weight_kg: float, height_cm: float) -> float:
    a, b, c, d = load_registry().value("hall.initial_body_fat", sex)
    bmi = weight_kg / (height_cm / 100) ** 2
    bf_pct = a * bmi + b * age + c * (1 if sex == "male" else 0) + d
    return weight_kg * float(np.clip(bf_pct, 5.0, 60.0)) / 100


def _derivatives(fat: float, lean: float, imbalance_mj: float, bp: BodyParams) -> tuple[float, float]:
    p = bp.c / (bp.c + fat)
    return (1 - p) * imbalance_mj / bp.rho_fat, p * imbalance_mj / bp.rho_lean


def simulate_weight(
    sex: str,
    age: float,
    height_cm: float,
    weight_kg: float,
    delta_intake_kcal: float,
    n_years: int,
    dt_days: float = 7.0,
) -> WeightTrajectory:
    """Integrate the model (RK2) and report body composition at each year."""
    bp = body_params(sex)
    fat0 = initial_fat_kg(sex, age, weight_kg, height_cm)
    lean0 = weight_kg - fat0
    delta_mj = delta_intake_kcal * KCAL_TO_MJ

    def imbalance(fat: float, lean: float) -> float:
        return delta_mj - bp.ee_lean * (lean - lean0) - bp.ee_fat * (fat - fat0)

    steps_per_year = max(1, int(round(DAYS_PER_YEAR / dt_days)))
    h = DAYS_PER_YEAR / steps_per_year
    fat, lean = fat0, lean0
    fats, leans = [fat], [lean]
    for _ in range(n_years):
        for _ in range(steps_per_year):
            k1 = _derivatives(fat, lean, imbalance(fat, lean), bp)
            mid_f, mid_l = fat + 0.5 * h * k1[0], lean + 0.5 * h * k1[1]
            k2 = _derivatives(mid_f, mid_l, imbalance(mid_f, mid_l), bp)
            fat = max(fat + h * k2[0], 0.5)
            lean = lean + h * k2[1]
        fats.append(fat)
        leans.append(lean)
    fat_arr, lean_arr = np.array(fats), np.array(leans)
    return WeightTrajectory(
        years=np.arange(n_years + 1),
        weight_kg=fat_arr + lean_arr,
        fat_kg=fat_arr,
        lean_kg=lean_arr,
        delta_intake_kcal=delta_intake_kcal,
    )


def steady_state_weight(sex: str, age: float, height_cm: float, weight_kg: float, delta_intake_kcal: float) -> float:
    """Weight where imbalance returns to zero along the Forbes path."""
    bp = body_params(sex)
    fat0 = initial_fat_kg(sex, age, weight_kg, height_cm)
    lean0 = weight_kg - fat0
    target_mj = delta_intake_kcal * KCAL_TO_MJ

    def gap(fat: float) -> float:
        lean = lean0 + bp.forbes * np.log(fat / fat0)
        return bp.ee_lean * (lean - lean0) + bp.ee_fat * (fat - fat0) - target_mj

    lo, hi = 0.5, 400.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    fat = (lo + hi) / 2
    return fat + lean0 + bp.forbes * np.log(fat / fat0)


def intake_delta_for_target_weight(sex: str, age: float, height_cm: float, weight_kg: float, target_kg: float) -> float:
    """Permanent intake change (kcal/d) whose steady state is target_kg."""
    bp = body_params(sex)
    fat0 = initial_fat_kg(sex, age, weight_kg, height_cm)
    lean0 = weight_kg - fat0

    def mass(fat: float) -> float:
        return fat + lean0 + bp.forbes * np.log(fat / fat0)

    lo, hi = 0.5, 400.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mass(mid) < target_kg else (lo, mid)
    fat = (lo + hi) / 2
    lean = lean0 + bp.forbes * np.log(fat / fat0)
    delta_mj = bp.ee_lean * (lean - lean0) + bp.ee_fat * (fat - fat0)
    return delta_mj / KCAL_TO_MJ
