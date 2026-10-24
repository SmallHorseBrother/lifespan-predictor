"""Calibration anchor checks against published Chinese results.

Both anchors are lag-free comparisons (the papers do not model lag), so the
checks use point coefficients and tau = 0.

1. Sun 2025 Nat Commun: population-wide ideal / practical scenarios applied
   to the CNHS-like reference population -> gain in period LE at age 30.
2. Sun 2022 Lancet Public Health: members of the reference population with
   all five CKB low-risk factors vs those with 0-1 -> difference in LE at 30.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exposures import ALCOHOL, SMOKING
from .hazards import point_coefficients
from .life_table import baseline_mx, life_expectancy
from .population import ReferencePopulation, mean_rr_by_age, population_log_hr, reference_population
from .registry import load_registry

_FORMER = SMOKING.index("former")
_CURRENT = (SMOKING.index("current_lt20"), SMOKING.index("current_ge20"))
_EXCESS = (ALCOHOL.index("daily_30_59"), ALCOHOL.index("daily_ge60"))
_MODERATE = ALCOHOL.index("daily_1_29")


# ---------- Nat Commun 2025 population scenarios ----------

def _quit_all(pop: ReferencePopulation) -> np.ndarray:
    return np.where(np.isin(pop.smoking, _CURRENT), _FORMER, pop.smoking)


def _quit_to_prevalence(pop: ReferencePopulation, target: float) -> np.ndarray:
    """Recode current smokers to former until each bucket's prevalence is `target`."""
    rng = np.random.default_rng(7)
    smoking = pop.smoking.copy()
    for b in np.unique(pop.bucket):
        members = np.flatnonzero((pop.bucket == b) & np.isin(pop.smoking, _CURRENT))
        share = members.size / np.count_nonzero(pop.bucket == b)
        if share > target:
            n_quit = int(round(members.size * (1 - target / share)))
            smoking[rng.choice(members, size=n_quit, replace=False)] = _FORMER
    return smoking


def _pa_plus(pop: ReferencePopulation, under65: float, over65: float) -> np.ndarray:
    return pop.pa_met_h + np.where(pop.age < 65, under65, over65)


def nc2025_population(pop: ReferencePopulation, scenario: str) -> ReferencePopulation:
    if scenario == "ideal":
        return pop.with_changes(
            smoking=_quit_all(pop),
            alcohol=np.where(np.isin(pop.alcohol, _EXCESS), _MODERATE, pop.alcohol),
            pa_met_h=_pa_plus(pop, 4.0, 2.0),
            fruit_g=pop.fruit_g + 200.0,
            fish_g=pop.fish_g + 50.0,
        )
    if scenario == "practical":
        smoking = _quit_to_prevalence(pop, 0.20) if pop.sex == "male" else pop.smoking
        return pop.with_changes(
            smoking=smoking,
            pa_met_h=_pa_plus(pop, 2.0, 1.0),
            fruit_g=pop.fruit_g + 100.0,
            fish_g=pop.fish_g + 20.0,
        )
    single = {
        "ideal_smoking": lambda: pop.with_changes(smoking=_quit_all(pop)),
        "ideal_alcohol": lambda: pop.with_changes(
            alcohol=np.where(np.isin(pop.alcohol, _EXCESS), _MODERATE, pop.alcohol)
        ),
        "ideal_pa": lambda: pop.with_changes(pa_met_h=_pa_plus(pop, 4.0, 2.0)),
        "ideal_diet": lambda: pop.with_changes(fruit_g=pop.fruit_g + 200.0, fish_g=pop.fish_g + 50.0),
    }
    if scenario not in single:
        raise ValueError(f"unknown NC2025 scenario {scenario}")
    return single[scenario]()


def nc2025_gain(sex: str, scenario: str) -> float:
    """Gain in period LE at age 30 when the scenario is applied to everyone."""
    c = point_coefficients(sex)
    pop = reference_population(sex)
    base_rr = mean_rr_by_age(pop, c)[0]
    scen_rr = mean_rr_by_age(nc2025_population(pop, scenario), c)[0]
    mx = baseline_mx(sex)
    return life_expectancy(mx * scen_rr / base_rr, 30) - life_expectancy(mx, 30)


# ---------- Lancet Public Health 2022 low-risk groups ----------

# Approximations of the CKB 2022 low-risk definitions on the reference
# population (估算 where the paper's variables are not in our model):
# - smoking: not a current smoker (cannot tell illness-related quitting)
# - alcohol: not drinking >=30 g/d daily (paper: <30 g men, <15 g women)
# - activity: >= median of the same sex in age bands <50, 50-59, >=60
# - diet: score 4-5 needs fruit daily and fish weekly in practice, approximated
#   by fruit >= 100 g/d and fish >= 20 g/d (CKB usual intake of the
#   4-6 d/wk fruit group is 79 g, daily 123 g; fish: monthly 19.7 g,
#   1-3 d/wk 31.3 g, so >= 20 g means eating fish at least weekly)
# - body shape: BMI 18.5-27.9 (waist circumference is not modelled)
DIET_FRUIT_MIN_G = 100.0
DIET_FISH_MIN_G = 20.0
PA_AGE_BANDS = (50, 60)


def low_risk_count(pop: ReferencePopulation) -> np.ndarray:
    smoking_ok = ~np.isin(pop.smoking, _CURRENT)
    alcohol_ok = ~np.isin(pop.alcohol, _EXCESS)
    band = np.digitize(pop.age, PA_AGE_BANDS)
    medians = np.array([np.median(pop.pa_met_h[band == b]) for b in range(len(PA_AGE_BANDS) + 1)])
    pa_ok = pop.pa_met_h >= medians[band]
    diet_ok = (pop.fruit_g >= DIET_FRUIT_MIN_G) & (pop.fish_g >= DIET_FISH_MIN_G)
    body_ok = (pop.bmi >= 18.5) & (pop.bmi < 28.0)
    return smoking_ok.astype(int) + alcohol_ok + pa_ok + diet_ok + body_ok


def _group_le30(sex: str, pop: ReferencePopulation, members: np.ndarray, base_rr: np.ndarray) -> float:
    """LE at 30 for a subgroup: m(a) = m_pop(a) * mean RR_group(a) / mean RR_pop(a)."""
    c = point_coefficients(sex)
    group = ReferencePopulation(
        sex=sex, **{k: getattr(pop, k)[members] for k in ReferencePopulation.__dataclass_fields__ if k != "sex"}
    )
    group_rr = mean_rr_by_age(group, c)[0]
    return life_expectancy(baseline_mx(sex) * group_rr / base_rr, 30)


ANCHOR_N_PER_BUCKET = 50_000  # 0-1 low-risk women are ~0.3% of the population


def ckb2022_comparison(sex: str) -> dict[str, float]:
    pop = reference_population(sex, ANCHOR_N_PER_BUCKET)
    c = point_coefficients(sex)
    base_rr = mean_rr_by_age(pop, c)[0]
    count = low_risk_count(pop)
    all5, low01 = count == 5, count <= 1
    le_all5 = _group_le30(sex, pop, all5, base_rr)
    le_low01 = _group_le30(sex, pop, low01, base_rr)
    rr = np.exp(population_log_hr(pop, c))[0]
    return {
        "share_all5": float(all5.mean()),
        "share_0_1": float(low01.mean()),
        "le30_all5": le_all5,
        "le30_0_1": le_low01,
        "gain": le_all5 - le_low01,
        "rr_all5_vs_0_1": float(rr[all5].mean() / rr[low01].mean()),
    }


def anchor_report() -> dict:
    reg = load_registry()
    out = {"nc2025": {}, "ckb2022": {}, "baseline": {}}
    for sex in ("male", "female"):
        targets = reg.value("anchor.nc2025", sex)
        out["nc2025"][sex] = {
            name: {"model": nc2025_gain(sex, name), "paper": targets[name]}
            for name in ("ideal", "practical", "ideal_smoking", "ideal_alcohol", "ideal_pa", "ideal_diet")
        }
        ckb = ckb2022_comparison(sex)
        ckb_target = reg.value("anchor.ckb2022", sex)
        out["ckb2022"][sex] = {**ckb, "paper_gain": ckb_target["le_gain"], "paper_hr": ckb_target["hr_5_vs_0_1"]}
        mx = baseline_mx(sex)
        out["baseline"][sex] = {
            "e0": life_expectancy(mx, 0),
            "e0_official": reg.value("baseline.official_e0_2020", sex),
            "e30": life_expectancy(mx, 30),
            "e30_nc2025_base": targets["base_le30"],
        }
    return out


__all__ = ["anchor_report", "nc2025_gain", "ckb2022_comparison"]
