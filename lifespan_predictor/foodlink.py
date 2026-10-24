"""FoodLink-facing output: LifespanResult and "run one life".

Field names follow the FoodLink design doc (food_link
docs/algorithms/foodlink-lifespan-simulator-v1-design.md, section 4.2).

- Branch curves, life-expectancy quantiles and histograms are computed by
  integrating the hazard with point estimates: deterministic, no Monte
  Carlo noise, so a Go port can match them to 1e-3.
- top_changes ranges come from parameter draws (numpy), for reference only.
- One life uses a portable SplitMix64 stream; all branches share the same
  hidden threshold (common random numbers).

Not available yet (returned as null with a warning): disease-free survival,
healthy years, disease events, input completeness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np

from .exposures import CKB_BASELINE_AGE, Profile, evidence_warnings, validate
from .hazards import draw_coefficients, point_coefficients
from .population import mean_rr_by_age, reference_population
from .registry import REGISTRY_PATH, load_registry
from .scenarios import Scenario, build_scenarios
from .simulate import _death_times, _expected_remaining, scenario_hazard

ENGINE_VERSION = "lifespan-cn-0.1.0"
HISTOGRAM_BIN_YEARS = 2
UINT64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15

# factor -> (Scenario fields it changes, label, unit, evidence badges, registry ids)
FACTORS = {
    "fruit": (("fruit_g",), "多吃水果", "g/d", ["CN Cohort"], ["diet.fruit_per_g", "diet.effect_caps"]),
    "fish": (("fish_g",), "多吃鱼和海产品", "g/d", ["CN Cohort"], ["diet.fish_per_g", "diet.effect_caps"]),
    "energy_balance": (
        ("delta_intake_kcal",),
        "体重回到合理区间",
        "kcal/d",
        ["Meta-analysis", "Mechanistic"],
        ["bmi.per5_above25_east_asia", "bmi.18_5_to_20", "bmi.15_to_18_5", "hall.energy_expenditure"],
    ),
    "smoking": (("smoking",), "戒烟", "-", ["CN Cohort"], ["smoking.former", "smoking.current_lt20", "smoking.current_ge20"]),
    "alcohol": (("alcohol",), "不每天喝酒", "-", ["CN Cohort"], ["alcohol.daily_1_29", "alcohol.daily_30_59", "alcohol.daily_ge60"]),
    "physical_activity": (("pa_met_h",), "多动", "MET-h/d", ["CN Cohort"], ["pa.ln_met_h", "pa.floor"]),
    "sleep": (("sleep_h",), "睡够 7-8 小时", "h", ["Meta-analysis"], ["sleep.short", "sleep.long", "sleep.shrinkage"]),
}
MODE_FACTORS = {
    "diet": ("fruit", "fish", "energy_balance"),
    "lifestyle": ("fruit", "fish", "energy_balance", "smoking", "alcohol", "physical_activity", "sleep"),
}


# ---------- portable random numbers ----------

def splitmix64(x: int) -> int:
    """One SplitMix64 output for state x (Steele, Lea & Flood 2014)."""
    z = (x + GOLDEN_GAMMA) & UINT64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64
    return z ^ (z >> 31)


def life_uniform(seed: int, life_index: int) -> float:
    """U(0,1) for one life: splitmix64(seed * GOLDEN_GAMMA + life_index), top 53 bits.

    The branch is deliberately not part of the derivation, so every branch of
    the same life sees the same number.
    """
    state = (seed * GOLDEN_GAMMA + life_index) & UINT64
    return ((splitmix64(state) >> 11) + 0.5) * 2.0**-53


# ---------- hashes ----------

def params_sha256() -> str:
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


def input_hash(profile: Profile, mode: str, lag_tau: float, n_param_draws: int) -> str:
    payload = {"profile": asdict(profile), "mode": mode, "lag_tau_years": lag_tau, "n_param_draws": n_param_draws}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- deterministic curve maths ----------

def _survival(hazard_row: np.ndarray) -> np.ndarray:
    return np.exp(-np.concatenate([[0.0], np.cumsum(hazard_row)]))


def _age_at_survival(start_age: float, hazard_row: np.ndarray, surv: np.ndarray, q: float) -> float:
    """Exact age where S = q under piecewise-constant hazard."""
    k = int(np.searchsorted(-surv, -q, side="right")) - 1
    if k >= hazard_row.size:
        return start_age + hazard_row.size
    return float(start_age + k + math.log(surv[k] / q) / hazard_row[k])


def _branch_curves(profile: Profile, hazard_row: np.ndarray, weight: np.ndarray, bmi: np.ndarray) -> dict:
    a0 = profile.age
    surv = _survival(hazard_row)
    ages = a0 + np.arange(surv.size)
    mean = a0 + float(_expected_remaining(hazard_row[None, :])[0])
    bins = np.arange(0, hazard_row.size + HISTOGRAM_BIN_YEARS, HISTOGRAM_BIN_YEARS)
    bins = np.minimum(bins, hazard_row.size)
    mass = surv[bins[:-1]] - surv[bins[1:]]
    return {
        "life_expectancy": {
            "median": round(_age_at_survival(a0, hazard_row, surv, 0.5), 2),
            "mean": round(mean, 2),
            "p10": round(_age_at_survival(a0, hazard_row, surv, 0.9), 2),
            "p90": round(_age_at_survival(a0, hazard_row, surv, 0.1), 2),
        },
        "healthy_years": None,
        "survival": [
            {"age": round(float(a), 2), "p_alive": round(float(s), 6), "p_alive_disease_free": None}
            for a, s in zip(ages, surv)
        ],
        "trajectory": [
            {"age": round(float(a), 2), "weight_kg": round(float(w), 2), "bmi": round(float(b), 2)}
            for a, w, b in zip(ages, weight, bmi)
        ],
        "death_age_histogram": {
            "bin_years": HISTOGRAM_BIN_YEARS,
            "start_age": round(a0, 2),
            "density": [round(float(m), 6) for m in mass],
            "note": "每个分箱的死亡概率，合计约为 1",
        },
    }


def _p_alive_at(profile: Profile, hazard_row: np.ndarray, age: float) -> float | None:
    offset = age - profile.age
    if offset <= 0 or offset > hazard_row.size:
        return None
    k = int(offset)
    surv = _survival(hazard_row)
    return float(surv[k] * math.exp(-hazard_row[min(k, hazard_row.size - 1)] * (offset - k)))


def _policy(scn: Scenario) -> dict:
    return {
        "smoking": scn.smoking,
        "alcohol": scn.alcohol,
        "pa_met_h": float(scn.pa_met_h),
        "fruit_g": float(scn.fruit_g),
        "fish_g": float(scn.fish_g),
        "sleep_h": float(scn.sleep_h),
        "energy_balance_kcal": round(float(scn.delta_intake_kcal), 1),
    }


# ---------- top changes ----------

def _plain(value):
    return value if isinstance(value, str) else round(float(value), 2)


def _top_changes(profile, mode, base, best, lag, n_param_draws, seed) -> list[dict]:
    rng = np.random.default_rng([seed, 3])
    c_draws = draw_coefficients(profile.sex, n_param_draws, rng)
    rr_draws = mean_rr_by_age(reference_population(profile.sex), c_draws)
    c_point = point_coefficients(profile.sex)
    rr_point = mean_rr_by_age(reference_population(profile.sex), c_point)
    base_draw = _expected_remaining(scenario_hazard(profile, base, base, c_draws, rr_draws, lag)[0])
    base_point = scenario_hazard(profile, base, base, c_point, rr_point, lag)[0][0]
    out = []
    for factor in MODE_FACTORS[mode]:
        fields, label, unit, evidence, pids = FACTORS[factor]
        target = {f: getattr(best, f) for f in fields}
        if all(getattr(base, f) == v for f, v in target.items()):
            continue
        single = replace(base, name=f"only_{factor}", **target)
        gains = _expected_remaining(scenario_hazard(profile, single, base, c_draws, rr_draws, lag)[0]) - base_draw
        point_row = scenario_hazard(profile, single, base, c_point, rr_point, lag)[0][0]
        p85 = (_p_alive_at(profile, point_row, 85.0), _p_alive_at(profile, base_point, 85.0))
        field = fields[0]
        out.append(
            {
                "factor": factor,
                "label": label,
                "current": _plain(getattr(base, field)),
                "target": _plain(target[field]),
                "unit": unit,
                "gain_years": {
                    "median": round(float(np.median(gains)), 2),
                    "low": round(float(np.percentile(gains, 2.5)), 2),
                    "high": round(float(np.percentile(gains, 97.5)), 2),
                },
                "gain_p85_pct": None if p85[0] is None else round((p85[0] - p85[1]) * 100, 2),
                "evidence": evidence,
                "parameter_ids": pids,
            }
        )
    return sorted(out, key=lambda x: -x["gain_years"]["median"])


# ---------- public API ----------

def lifespan_result(
    profile: Profile,
    mode: str = "diet",
    seed: int = 20261024,
    n_param_draws: int = 200,
    lag_tau_years: float | None = None,
) -> dict:
    """LifespanResult for FoodLink. mode "diet" (default) or "lifestyle"."""
    validate(profile)
    reg = load_registry()
    lag = float(reg.value("lag.effect_tau_years", profile.sex)) if lag_tau_years is None else lag_tau_years
    futures = build_scenarios(profile, mode)
    base = futures[0]
    c = point_coefficients(profile.sex)
    mean_rr = mean_rr_by_age(reference_population(profile.sex), c)

    branches, warnings = [], []
    for scn in futures:
        hazard, weight, bmi = scenario_hazard(profile, scn, base, c, mean_rr, lag)
        branch = {"key": scn.name, "label": scn.label, "policy": _policy(scn)}
        branch.update(_branch_curves(profile, hazard[0], weight, bmi))
        branches.append(branch)
        for w in evidence_warnings(
            profile.sex, profile.age, scn.pa_met_h, scn.fruit_g, scn.fish_g, scn.sleep_h, float(bmi.max())
        ):
            warnings.append({**w, "branch": scn.name})
    warnings.append(
        {
            "code": "NOT_MODELLED",
            "field": "survival.p_alive_disease_free",
            "message": "第一版没有疾病模块，无病生存、健康年数和疾病事件为 null",
        }
    )
    current_fields = sorted({w["field"] for w in warnings if w.get("branch") == base.name and w["code"] == "EXTRAPOLATED"})
    return {
        "model": {
            "engine_version": ENGINE_VERSION,
            "params_version": reg.version,
            "params_sha256": params_sha256(),
            "input_hash": input_hash(profile, mode, lag, n_param_draws),
            "seed": seed,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "mode": mode,
        "disclaimer": "模拟估算，不用于诊断",
        "applicability": {
            "age_in_range": CKB_BASELINE_AGE[0] <= profile.age <= CKB_BASELINE_AGE[1],
            "extrapolated_inputs": current_fields,
            "completeness_pct": None,
        },
        "branches": branches,
        "top_changes": _top_changes(profile, mode, base, futures[1], lag, n_param_draws, seed),
        "top_changes_note": "单因素反事实：每次只把一项改到目标值，各项收益合计不等于全部一起改的收益",
        "warnings": warnings,
    }


def one_life(profile: Profile, mode: str = "diet", seed: int = 20261024, life_index: int = 0, lag_tau_years: float | None = None) -> dict:
    """One random life per branch, all branches sharing one SplitMix64 threshold."""
    validate(profile)
    lag = float(load_registry().value("lag.effect_tau_years", profile.sex)) if lag_tau_years is None else lag_tau_years
    futures = build_scenarios(profile, mode)
    c = point_coefficients(profile.sex)
    mean_rr = mean_rr_by_age(reference_population(profile.sex), c)
    threshold = np.array([[-math.log(1.0 - life_uniform(seed, life_index))]])
    branches = []
    for scn in futures:
        hazard = scenario_hazard(profile, scn, futures[0], c, mean_rr, lag)[0]
        death_age = round(profile.age + float(_death_times(hazard, threshold)[0, 0]), 2)
        branches.append({"key": scn.name, "death_age": death_age, "events": [{"age": death_age, "type": "death"}]})
    return {
        "seed": seed,
        "life_index": life_index,
        "note": "一次随机结果，不是预测。第一版只有死亡事件，没有疾病事件",
        "branches": branches,
    }
