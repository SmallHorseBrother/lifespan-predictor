"""Monte Carlo lifespan simulation with common random numbers.

Nested design: D parameter draws (published coefficients sampled from their
95% CIs) x L lives per draw. Every life j has one hidden Exp(1) threshold E_j
and one parameter draw; all scenarios reuse both, so a life dies in each
future when its cumulative hazard crosses the same E_j. Differences between
futures therefore come from the hazards, not from different dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exposures import Profile, evidence_flags, validate
from .hazards import (
    Coefficients,
    age_attenuation,
    alcohol_code,
    behaviour_log_hr,
    bmi_log_hr,
    draw_coefficients,
    point_coefficients,
    smoking_code,
)
from .life_table import MAX_AGE, baseline_mx
from .population import mean_rr_by_age, reference_population
from .registry import load_registry
from .scenarios import Scenario, build_scenarios
from .weight import simulate_weight

MODEL_VERSION = "lifespan-predictor 0.1.0"
REACH_AGES = (70, 80, 90, 100)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    death_age: np.ndarray  # (n_lives,)
    ages: np.ndarray  # integer-offset ages a0, a0+1, ..., MAX_AGE
    survival: np.ndarray  # empirical P(alive) at `ages`
    weight_kg: np.ndarray  # body weight at `ages`
    bmi: np.ndarray
    expected_remaining_by_draw: np.ndarray  # (D,) analytic, for model uncertainty
    flags: list[str]

    def summary(self, start_age: float) -> dict:
        remaining = self.death_age - start_age
        q = np.percentile(remaining, [10, 25, 50, 75, 90])
        model = self.expected_remaining_by_draw
        out = {
            "scenario": self.scenario.name,
            "label": self.scenario.label,
            "remaining_years": {
                "mean": float(remaining.mean()),
                "p10": float(q[0]),
                "p25": float(q[1]),
                "median": float(q[2]),
                "p75": float(q[3]),
                "p90": float(q[4]),
            },
            "expected_remaining_model_interval": [
                float(np.percentile(model, 2.5)),
                float(np.percentile(model, 97.5)),
            ],
            "p_reach_age": {str(a): float((self.death_age >= a).mean()) for a in REACH_AGES if a > start_age},
            "weight_kg_start_end": [float(self.weight_kg[0]), float(self.weight_kg[-1])],
            "inputs": {
                "smoking": self.scenario.smoking,
                "alcohol": self.scenario.alcohol,
                "pa_met_h": self.scenario.pa_met_h,
                "fruit_g": self.scenario.fruit_g,
                "fish_g": self.scenario.fish_g,
                "sleep_h": self.scenario.sleep_h,
                "delta_intake_kcal": round(self.scenario.delta_intake_kcal, 1),
            },
            "evidence_flags": self.flags,
        }
        return out


@dataclass(frozen=True)
class SimulationResult:
    profile: Profile
    results: dict[str, ScenarioResult]
    seed: int
    n_param_draws: int
    lag_tau_years: float
    model_version: str = MODEL_VERSION
    registry_version: str = field(default_factory=lambda: load_registry().version)

    def paired_difference(self, scenario: str, reference: str | None = None) -> dict:
        """Mean paired difference in death age (same lives, same parameter draws).

        Pairing removes Monte Carlo noise from the difference. It does not mean
        every individual is guaranteed to gain: with one threshold per life and
        uniformly lower hazard, paired lives are ordered by construction.
        """
        reference = reference or next(iter(self.results))
        diff = self.results[scenario].death_age - self.results[reference].death_age
        return {"mean_years": float(diff.mean())}

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "registry_version": self.registry_version,
            "seed": self.seed,
            "n_lives": int(next(iter(self.results.values())).death_age.size),
            "n_param_draws": self.n_param_draws,
            "lag_tau_years": self.lag_tau_years,
            "disclaimer": "模拟估算，不用于诊断",
            "profile": self.profile.__dict__,
            "scenarios": [r.summary(self.profile.age) for r in self.results.values()],
            "paired_vs_reference": {
                name: self.paired_difference(name) for name in list(self.results)[1:]
            },
        }


def _behaviour_lhr(scn: Scenario, c: Coefficients) -> np.ndarray:
    return behaviour_log_hr(
        np.array([smoking_code(scn.smoking)]),
        np.array([alcohol_code(scn.alcohol)]),
        np.array([scn.pa_met_h]),
        np.array([scn.fruit_g]),
        np.array([scn.fish_g]),
        np.array([scn.sleep_h]),
        c,
    )  # (D, 1)


def scenario_hazard(
    profile: Profile,
    scn: Scenario,
    base_scn: Scenario,
    c: Coefficients,
    mean_rr: np.ndarray,
    lag_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annual hazards (D, n_years) plus yearly weight and BMI (n_years + 1,).

    Interval k covers attained ages [age + k, age + k + 1). Single-age rates
    m_x are treated as values at age x + 0.5 and interpolated log-linearly to
    the interval midpoint, so results change smoothly with the starting age.
    """
    n_years = int(MAX_AGE - profile.age)
    traj = simulate_weight(
        profile.sex, profile.age, profile.height_cm, profile.weight_kg, scn.delta_intake_kcal, n_years
    )
    height_m2 = (profile.height_cm / 100) ** 2
    bmi = traj.weight_kg / height_m2
    bmi_mid = (bmi[:-1] + bmi[1:]) / 2

    old = _behaviour_lhr(base_scn, c)
    new = _behaviour_lhr(scn, c)
    k = np.arange(n_years) + 0.5
    remaining = np.exp(-k / lag_tau) if lag_tau > 0 else np.zeros(n_years)
    behaviour = new + (old - new) * remaining[None, :]

    mid_age = profile.age + k
    log_hr = (behaviour + bmi_log_hr(bmi_mid[None, :], c)) * age_attenuation(profile.sex, mid_age)[None, :]
    base = interpolate_log(baseline_mx(profile.sex)[None, :], mid_age)
    hazard = base * np.exp(log_hr) / interpolate_log(mean_rr, mid_age)
    return hazard, traj.weight_kg, bmi


def interpolate_log(values: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Log-linear interpolation of single-age values (D, MAX_AGE+1) at ages.

    values[:, x] is taken to apply at age x + 0.5; outside the knots the end
    values are held.
    """
    pos = np.clip(np.asarray(ages, dtype=float) - 0.5, 0.0, MAX_AGE)
    lo = np.minimum(np.floor(pos).astype(int), MAX_AGE - 1)
    w = pos - lo
    logv = np.log(values)
    return np.exp((1 - w) * logv[:, lo] + w * logv[:, lo + 1])


def _death_times(hazard: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """First time cumulative piecewise-constant hazard exceeds each threshold."""
    n_draws, n_years = hazard.shape
    cum = np.cumsum(hazard, axis=1)
    times = np.empty_like(thresholds)
    for d in range(n_draws):
        idx = np.searchsorted(cum[d], thresholds[d], side="right")
        idx_c = np.minimum(idx, n_years - 1)
        prev = np.where(idx_c > 0, cum[d][idx_c - 1], 0.0)
        t = idx_c + (thresholds[d] - prev) / hazard[d][idx_c]
        times[d] = np.where(idx >= n_years, float(n_years), t)
    return times


def _expected_remaining(hazard: np.ndarray) -> np.ndarray:
    """Analytic E[T] under piecewise-constant hazard, per draw."""
    surv_start = np.exp(-np.concatenate([np.zeros((hazard.shape[0], 1)), np.cumsum(hazard, axis=1)[:, :-1]], axis=1))
    return (surv_start * (1 - np.exp(-hazard)) / hazard).sum(axis=1)


def simulate(
    profile: Profile,
    n_lives: int = 10_000,
    n_param_draws: int = 100,
    seed: int = 20261024,
    lag_tau_years: float | None = None,
    parameter_uncertainty: bool = True,
    scenarios: tuple[Scenario, ...] | None = None,
) -> SimulationResult:
    """Simulate the forked futures of one person.

    scenarios: optional custom futures (e.g. diet-only forks); the first one
    is the reference that lagged effects start from and pairing compares to.
    Default: current / optimized / adverse.
    n_lives must be a multiple of the number of parameter draws.
    """
    validate(profile)
    if n_lives < 1 or n_param_draws < 1:
        raise ValueError("n_lives and n_param_draws must be positive")
    lag = float(load_registry().value("lag.effect_tau_years", profile.sex)) if lag_tau_years is None else lag_tau_years
    if lag < 0:
        raise ValueError("lag_tau_years must be >= 0")

    draws = min(n_param_draws, n_lives) if parameter_uncertainty else 1
    if n_lives % draws:
        raise ValueError(f"n_lives ({n_lives}) must be a multiple of n_param_draws ({draws})")
    lives_per_draw = n_lives // draws
    coef_rng = np.random.default_rng([seed, 1])
    life_rng = np.random.default_rng([seed, 2])
    c = draw_coefficients(profile.sex, draws, coef_rng) if parameter_uncertainty else point_coefficients(profile.sex)
    mean_rr = mean_rr_by_age(reference_population(profile.sex), c)
    thresholds = life_rng.exponential(size=(draws, lives_per_draw))

    futures = build_scenarios(profile) if scenarios is None else tuple(scenarios)
    if not futures:
        raise ValueError("scenarios must not be empty")
    base = futures[0]
    results = {}
    for scn in futures:
        hazard, weight, bmi = scenario_hazard(profile, scn, base, c, mean_rr, lag)
        t = _death_times(hazard, thresholds).ravel()
        death_age = profile.age + t
        ages = profile.age + np.arange(hazard.shape[1] + 1)
        survival = (death_age[None, :] > ages[:, None]).mean(axis=1)
        flags = evidence_flags(
            profile.sex, profile.age, scn.pa_met_h, scn.fruit_g, scn.fish_g, scn.sleep_h, float(bmi.max())
        )
        if bmi.min() < 15:
            flags.append(f"BMI 最低降到 {bmi.min():.1f}，风险为外推")
        results[scn.name] = ScenarioResult(
            scenario=scn,
            death_age=death_age,
            ages=ages,
            survival=survival,
            weight_kg=weight,
            bmi=bmi,
            expected_remaining_by_draw=_expected_remaining(hazard),
            flags=flags,
        )
    return SimulationResult(
        profile=profile, results=results, seed=seed, n_param_draws=draws, lag_tau_years=lag
    )
