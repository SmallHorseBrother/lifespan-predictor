"""Synthetic reference population built from CNHS 2015 published aggregates.

The baseline life table describes the whole Chinese population, which is a
mix of smokers, non-smokers, active and inactive people. Individual hazards
are therefore re-centred: m_i(a) = m_pop(a) * RR_i(a) / mean_RR_pop(a), so a
population with the CNHS exposure mix reproduces the life table exactly.

Only published summary numbers are used (Sun 2025 Supplementary Table 1 and
Source Data Figure S6); CKB/CNHS individual data are not used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .hazards import Coefficients, age_attenuation, behaviour_log_hr, bmi_log_hr
from .life_table import MAX_AGE
from .registry import load_registry

GROUP_EDGES = (30, 40, 50, 60, 70, 80)  # registry by-age arrays: 30-39 ... >=80
BUCKET_STARTS = tuple(range(30, 90, 5))  # 5-year buckets 30-34 ... 85+
OLDEST_BUCKET_END = 95
N_PER_BUCKET = 4000
SEED = 20251107

_BMI_BINS = ((16.0, 18.5), (18.5, 24.0), (24.0, 28.0), (28.0, 34.0))
_SLEEP_BINS = ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.5))


@dataclass(frozen=True)
class ReferencePopulation:
    sex: str
    bucket: np.ndarray  # index into BUCKET_STARTS
    age: np.ndarray
    smoking: np.ndarray  # SMOKING codes
    alcohol: np.ndarray  # ALCOHOL codes
    pa_met_h: np.ndarray
    fruit_g: np.ndarray
    fish_g: np.ndarray
    bmi: np.ndarray
    sleep_h: np.ndarray

    def with_changes(self, **changes: np.ndarray) -> "ReferencePopulation":
        return replace(self, **changes)


def bucket_of_age(age: np.ndarray | int) -> np.ndarray:
    a = np.asarray(age)
    return np.clip((a - BUCKET_STARTS[0]) // 5, 0, len(BUCKET_STARTS) - 1).astype(int)


def _group_of_bucket(bucket_start: int) -> int:
    return min((bucket_start - GROUP_EDGES[0]) // 10, len(GROUP_EDGES) - 1)


def _lognormal(rng: np.random.Generator, mean: float, cv: float, n: int) -> np.ndarray:
    sigma2 = np.log1p(cv**2)
    return rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2), size=n)


def _categorical_from_bins(rng: np.random.Generator, shares, bins, n: int) -> np.ndarray:
    p = np.asarray(shares, dtype=float)
    idx = rng.choice(len(bins), size=n, p=p / p.sum())
    lo = np.array([b[0] for b in bins])[idx]
    hi = np.array([b[1] for b in bins])[idx]
    return rng.uniform(lo, hi)


def _split_binary(rng, p_high: float, high_codes, high_shares, low_codes, low_shares, n: int):
    """Draw 'high' with probability p_high, then split each side by overall shares."""
    is_high = rng.random(n) < p_high
    high = rng.choice(high_codes, size=n, p=np.asarray(high_shares) / np.sum(high_shares))
    low = rng.choice(low_codes, size=n, p=np.asarray(low_shares) / np.sum(low_shares))
    return np.where(is_high, high, low)


@lru_cache(maxsize=4)
def reference_population(sex: str, n_per_bucket: int = N_PER_BUCKET) -> ReferencePopulation:
    """CNHS-like population; the default size is enough for population means.

    Anchor checks on small subgroups (e.g. women with 0-1 low-risk factors)
    use a larger n_per_bucket to keep sampling noise down.
    """
    reg = load_registry()
    rng = np.random.default_rng(SEED + (0 if sex == "male" else 1))
    smk = reg.value("pop.cnhs_smoking", sex)  # never, former, <20, >=20
    alc = reg.value("pop.cnhs_alcohol", sex)  # <daily, 1-29, 30-59, >=60
    sd = reg.value("pop.cnhs_overall_sd", sex)
    cv = {k: v[1] / v[0] for k, v in sd.items()}
    by_age = {
        "smoke": reg.value("pop.cnhs_current_smoking_by_age", sex),
        "alcohol": reg.value("pop.cnhs_excess_alcohol_by_age", sex),
        "pa": reg.value("pop.cnhs_pa_mean_by_age", sex),
        "fruit": reg.value("pop.cnhs_fruit_mean_by_age", sex),
        "fish": reg.value("pop.cnhs_fish_mean_by_age", sex),
    }
    bmi_shares = reg.value("pop.cnhs_bmi", sex)
    sleep_shares = reg.value("pop.sleep_distribution", sex)

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ReferencePopulation.__dataclass_fields__ if k != "sex"}
    n = n_per_bucket
    for b, start in enumerate(BUCKET_STARTS):
        g = _group_of_bucket(start)
        end = start + 5 if start < BUCKET_STARTS[-1] else OLDEST_BUCKET_END
        parts["bucket"].append(np.full(n, b))
        parts["age"].append(rng.uniform(start, end, size=n))
        parts["smoking"].append(
            _split_binary(rng, by_age["smoke"][g] / 100, [2, 3], smk[2:], [0, 1], smk[:2], n)
        )
        parts["alcohol"].append(
            _split_binary(rng, by_age["alcohol"][g] / 100, [2, 3], alc[2:], [0, 1], alc[:2], n)
        )
        parts["pa_met_h"].append(_lognormal(rng, by_age["pa"][g], cv["pa"], n))
        parts["fruit_g"].append(_lognormal(rng, by_age["fruit"][g], cv["fruit"], n))
        parts["fish_g"].append(_lognormal(rng, by_age["fish"][g], cv["fish"], n))
        parts["bmi"].append(_categorical_from_bins(rng, bmi_shares, _BMI_BINS, n))
        parts["sleep_h"].append(_categorical_from_bins(rng, sleep_shares, _SLEEP_BINS, n))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for arr in arrays.values():
        arr.setflags(write=False)
    return ReferencePopulation(sex=sex, **arrays)


def population_log_hr(pop: ReferencePopulation, c: Coefficients) -> np.ndarray:
    """Total log HR for every member, shape (D, N)."""
    return behaviour_log_hr(
        pop.smoking, pop.alcohol, pop.pa_met_h, pop.fruit_g, pop.fish_g, pop.sleep_h, c
    ) + bmi_log_hr(pop.bmi, c)


def mean_rr_by_age(pop: ReferencePopulation, c: Coefficients) -> np.ndarray:
    """Mean relative risk at single ages 0..MAX_AGE, shape (D, MAX_AGE + 1).

    Members of the 5-year bucket containing age a stand in for the population
    at age a; their log HRs are scaled by the age attenuation at a.
    """
    log_hr = population_log_hr(pop, c)
    ages = np.arange(MAX_AGE + 1)
    att = age_attenuation(pop.sex, ages)
    buckets = bucket_of_age(ages)
    out = np.empty((log_hr.shape[0], ages.size))
    for b in np.unique(buckets):
        member_lhr = log_hr[:, pop.bucket == b]
        for a in ages[buckets == b]:
            out[:, a] = np.exp(member_lhr * att[a]).mean(axis=1)
    return out
