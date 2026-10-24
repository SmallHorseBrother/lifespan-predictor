"""The three futures that fork from the same person.

Two modes:

lifestyle (多因素): every modelled factor may change.
    current   (照现在)   : today's habits continue unchanged.
    optimized (优化)     : the large, well-evidenced factors are fixed.
    adverse   (多重不良) : several realistic bad habits accumulate.

diet (只改饮食, FoodLink default): smoking, alcohol, activity and sleep stay
as today in all three futures; only fruit, fish and energy balance differ.
    current    (照现在这样吃)
    diet_right (把饮食吃对)
    diet_loose (放飞自我)

Targets come from the registry (scenario.*, status 估算): they are product
settings, never values tuned to produce a particular lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .exposures import ALCOHOL, SMOKING, Profile
from .registry import load_registry
from .weight import intake_delta_for_target_weight

LABELS = {
    "current": "照现在",
    "optimized": "优化",
    "adverse": "多重不良",
    "diet_right": "把饮食吃对",
    "diet_loose": "放飞自我",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    smoking: str
    alcohol: str
    pa_met_h: float
    fruit_g: float
    fish_g: float
    sleep_h: float
    delta_intake_kcal: float


def current(profile: Profile) -> Scenario:
    return Scenario(
        name="current",
        label=LABELS["current"],
        smoking=profile.smoking,
        alcohol=profile.alcohol,
        pa_met_h=profile.pa_met_h,
        fruit_g=profile.fruit_g,
        fish_g=profile.fish_g,
        sleep_h=profile.sleep_h,
        delta_intake_kcal=profile.energy_balance_kcal,
    )


def _intake_delta_to_bmi_range(profile: Profile, bmi_range) -> float:
    """Permanent intake change that moves BMI into bmi_range (0 if already inside)."""
    lo, hi = bmi_range
    target_bmi = min(max(profile.bmi, lo), hi)
    if target_bmi == profile.bmi:
        return 0.0
    target_kg = target_bmi * (profile.height_cm / 100) ** 2
    return intake_delta_for_target_weight(profile.sex, profile.age, profile.height_cm, profile.weight_kg, target_kg)


def optimized(profile: Profile) -> Scenario:
    t = load_registry().value("scenario.optimized_targets", profile.sex)
    delta = _intake_delta_to_bmi_range(profile, t["bmi_range"])
    return Scenario(
        name="optimized",
        label=LABELS["optimized"],
        smoking="never" if profile.smoking == "never" else "former",
        alcohol="less_than_daily",
        pa_met_h=max(profile.pa_met_h + t["pa_increment"], t["pa_minimum"]),
        fruit_g=max(profile.fruit_g, t["fruit_g"]),
        fish_g=max(profile.fish_g, t["fish_g"]),
        sleep_h=t["sleep_h"],
        delta_intake_kcal=delta,
    )


def adverse(profile: Profile) -> Scenario:
    reg = load_registry()
    t = reg.value("scenario.adverse_targets", profile.sex)
    surplus = float(reg.value("scenario.adverse_energy_surplus", profile.sex))
    return Scenario(
        name="adverse",
        label=LABELS["adverse"],
        smoking=SMOKING[max(SMOKING.index(profile.smoking), SMOKING.index(t["smoking"]))],
        alcohol=ALCOHOL[max(ALCOHOL.index(profile.alcohol), ALCOHOL.index(t["alcohol"]))],
        pa_met_h=min(profile.pa_met_h, t["pa_met_h"]),
        fruit_g=min(profile.fruit_g, t["fruit_g"]),
        fish_g=min(profile.fish_g, t["fish_g"]),
        sleep_h=min(profile.sleep_h, t["sleep_h"]),
        delta_intake_kcal=max(profile.energy_balance_kcal, surplus),
    )


def diet_right(profile: Profile) -> Scenario:
    t = load_registry().value("scenario.diet_right_targets", profile.sex)
    return replace(
        current(profile),
        name="diet_right",
        label=LABELS["diet_right"],
        fruit_g=max(profile.fruit_g, t["fruit_g"]),
        fish_g=max(profile.fish_g, t["fish_g"]),
        delta_intake_kcal=_intake_delta_to_bmi_range(profile, t["bmi_range"]),
    )


def diet_loose(profile: Profile) -> Scenario:
    reg = load_registry()
    t = reg.value("scenario.diet_loose_targets", profile.sex)
    surplus = float(reg.value("scenario.adverse_energy_surplus", profile.sex))
    return replace(
        current(profile),
        name="diet_loose",
        label=LABELS["diet_loose"],
        fruit_g=min(profile.fruit_g, t["fruit_g"]),
        fish_g=min(profile.fish_g, t["fish_g"]),
        delta_intake_kcal=max(profile.energy_balance_kcal, surplus),
    )


MODES = {
    "lifestyle": (current, optimized, adverse),
    "diet": (current, diet_right, diet_loose),
}


def build_scenarios(profile: Profile, mode: str = "lifestyle") -> tuple[Scenario, ...]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    return tuple(build(profile) for build in MODES[mode])
