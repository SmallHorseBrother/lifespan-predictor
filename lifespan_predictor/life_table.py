"""Baseline mortality: calibrated 2020 China life table (single ages 0-110)."""

from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np

from .registry import ROOT, SEXES, load_registry

MAX_AGE = 110


@lru_cache(maxsize=2)
def baseline_mx(sex: str) -> np.ndarray:
    """Central death rates m_x for ages 0..MAX_AGE (index = age)."""
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    rel_path = load_registry().value("baseline.life_table_cn2020", sex)
    rates = np.full(MAX_AGE + 1, np.nan)
    with (ROOT / rel_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sex"] == sex:
                rates[int(row["age"])] = float(row["mx"])
    if np.isnan(rates).any():
        raise ValueError(f"life table incomplete for {sex}")
    rates.setflags(write=False)
    return rates


def life_expectancy(mx: np.ndarray, from_age: int = 0) -> float:
    """Period life expectancy at `from_age` from single-age m_x.

    Uses a_x = 0.5 (0.1 at age 0) and closes the last age with 1/m.
    Must agree with scripts/build_life_table.py.
    """
    mx = np.asarray(mx, dtype=float)
    last = len(mx) - 1
    survivors, total = 1.0, 0.0
    for age in range(from_age, last):
        m = mx[age]
        ax = 0.1 if age == 0 else 0.5
        qx = min(m / (1 + (1 - ax) * m), 1.0)
        deaths = survivors * qx
        total += survivors - (1 - ax) * deaths
        survivors -= deaths
    return total + survivors / mx[last]


def survival_curve(mx: np.ndarray, from_age: int) -> np.ndarray:
    """S(age | alive at from_age) for ages from_age..MAX_AGE, using exp(-m)."""
    hazards = np.asarray(mx, dtype=float)[from_age:]
    return np.concatenate([[1.0], np.exp(-np.cumsum(hazards[:-1]))])
