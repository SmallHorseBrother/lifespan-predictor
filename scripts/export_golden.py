"""Export golden fixtures (input -> output) shared by the Python, Go and JS engines.

Each case fixes a profile, a mode, an optional custom future ("play"), a plan
(change_start_age, adherence) and a seed. Outputs are the deterministic
point-estimate results per branch, unrounded:

- life_expectancy {mean, median, p10, p90} (ages at death)
- p_alive at ages 40/60/70/80/90/100 that lie after the start age
- weight_kg {start, mid, end} of the trajectory
- one_life: death ages for life_index 0, 1, 2 (SplitMix64, shared by branches)

Tolerances for ports: survival probability <= 0.001, ages <= 0.05 years,
one_life <= 0.05 years.

Usage:
    python scripts/export_golden.py           # write golden/fixtures.json
    python scripts/export_golden.py --check   # exit 1 if outputs drifted
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifespan_predictor.exposures import ALCOHOL, SMOKING, Profile  # noqa: E402
from lifespan_predictor.foodlink import (  # noqa: E402
    ENGINE_VERSION,
    _age_at_survival,
    _p_alive_at,
    _survival,
    life_uniform,
)
from lifespan_predictor.hazards import point_coefficients  # noqa: E402
from lifespan_predictor.plan import make_plan  # noqa: E402
from lifespan_predictor.population import mean_rr_by_age, reference_population  # noqa: E402
from lifespan_predictor.registry import load_registry  # noqa: E402
from lifespan_predictor.scenarios import Scenario, build_scenarios  # noqa: E402
from lifespan_predictor.simulate import _death_times, _expected_remaining, scenario_hazard  # noqa: E402

OUT = ROOT / "golden" / "fixtures.json"
N_CASES = 240
P_ALIVE_AGES = (40, 60, 70, 80, 90, 100)
LIFE_INDEXES = (0, 1, 2)
TOLERANCE = {"age_years": 0.05, "p_alive": 0.001}

AGES = (18.0, 30.0, 50.0, 79.0, 85.0)
SEXES = ("male", "female")
BMIS = (16.0, 18.7, 22.0, 25.5, 29.0, 34.0, 40.0)
ADHERENCE = (None, 0.0, 0.5, {"type": "decay", "start": 0.9, "end": 0.5, "half_life_years": 2.0}, 0.7)
START_KINDS = ("now", "mid", "near_100")


def _start_age(kind: str, age: float) -> float | None:
    if kind == "now":
        return None
    if kind == "mid":
        return 45.0 if age < 45 else age + 10
    return 99.0


def case_inputs() -> list[dict]:
    rnd = random.Random(20261028)
    cases = []
    for i in range(N_CASES):
        age = AGES[i % len(AGES)] + (0.4 if i % 7 == 3 else 0.0)  # some fractional ages
        sex = SEXES[(i // len(AGES)) % 2]
        height = round(rnd.uniform(150, 190), 1)
        extrapolate = i % 6 == 5
        profile = {
            "sex": sex,
            "age": age,
            "height_cm": height,
            "weight_kg": round(BMIS[(i // 2) % len(BMIS)] * (height / 100) ** 2, 1),
            "smoking": SMOKING[(i // 10) % 4],
            "alcohol": ALCOHOL[(i // 40) % 4],
            "pa_met_h": round(rnd.choice([70.0, 1.0]) if extrapolate else rnd.uniform(3, 45), 1),
            "fruit_g": float(round(rnd.choice([500, 5]) if extrapolate else rnd.uniform(25, 250))),
            "fish_g": float(round(rnd.choice([150, 0]) if extrapolate else rnd.uniform(3, 60))),
            "sleep_h": rnd.choice([3.5, 12.0]) if extrapolate else rnd.choice([5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
            "energy_balance_kcal": float(round(rnd.uniform(-200, 300))) if i % 5 == 2 else 0.0,
        }
        custom = None
        if i % 4 == 3:
            custom = {
                "name": "play",
                "label": "play",
                "smoking": rnd.choice(SMOKING),
                "alcohol": rnd.choice(ALCOHOL),
                "pa_met_h": round(rnd.uniform(0, 60), 1),
                "fruit_g": float(round(rnd.uniform(0, 400))),
                "fish_g": float(round(rnd.uniform(0, 120))),
                "sleep_h": rnd.choice([4.0, 6.0, 7.0, 8.0, 9.0]),
                "delta_intake_kcal": float(round(rnd.uniform(-500, 500))),
            }
        cases.append(
            {
                "id": f"case-{i:03d}",
                "profile": profile,
                "mode": "diet" if i % 2 == 0 else "lifestyle",
                "custom_scenario": custom,
                "change_start_age": _start_age(START_KINDS[i % 3], age),
                "adherence": ADHERENCE[(i // 3) % len(ADHERENCE)],
                "lag_tau_years": None,
                "seed": 20261028 + i,
            }
        )
    return cases


def branch_output(profile: Profile, scn: Scenario, base: Scenario, plan, lag: float, seed: int) -> dict:
    c = point_coefficients(profile.sex)
    mean_rr = mean_rr_by_age(reference_population(profile.sex), c)
    hazard, weight, _ = scenario_hazard(profile, scn, base, c, mean_rr, lag, plan)
    row = hazard[0]
    surv = _survival(row)
    thresholds = np.array([[-math.log(1.0 - life_uniform(seed, j)) for j in LIFE_INDEXES]])
    deaths = profile.age + _death_times(hazard, thresholds)[0]
    return {
        "scenario": {k: v for k, v in asdict(scn).items() if k != "label"},
        "life_expectancy": {
            "mean": profile.age + float(_expected_remaining(hazard)[0]),
            "median": _age_at_survival(profile.age, row, surv, 0.5),
            "p10": _age_at_survival(profile.age, row, surv, 0.9),
            "p90": _age_at_survival(profile.age, row, surv, 0.1),
        },
        "p_alive": {str(a): _p_alive_at(profile, row, float(a)) for a in P_ALIVE_AGES if a > profile.age},
        "weight_kg": {
            "start": float(weight[0]),
            "mid": float(weight[weight.size // 2]),
            "end": float(weight[-1]),
        },
        "one_life": [float(d) for d in deaths],
    }


def case_outputs(case: dict) -> dict:
    profile = Profile(**case["profile"])
    plan = make_plan(case["change_start_age"], case["adherence"]).validate(profile.age)
    lag = case["lag_tau_years"]
    lag = float(load_registry().value("lag.effect_tau_years", profile.sex)) if lag is None else lag
    futures = list(build_scenarios(profile, case["mode"]))
    if case["custom_scenario"]:
        futures.append(Scenario(**case["custom_scenario"]))
    base = futures[0]
    return {scn.name: branch_output(profile, scn, base, plan, lag, case["seed"]) for scn in futures}


def _rounded(obj):
    if isinstance(obj, float):
        return round(obj, 9)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    return obj


def build() -> dict:
    cases = []
    for case in case_inputs():
        cases.append({**case, "expected": _rounded(case_outputs(case))})
    return {
        "engine_version": ENGINE_VERSION,
        "registry_version": load_registry().version,
        "tolerance": TOLERANCE,
        "notes": (
            "change_start_age null = start now; adherence null = 1; lag_tau_years null = registry default. "
            "Plan applies to every branch except the first (reference). Values are unrounded point estimates."
        ),
        "cases": cases,
    }


def main() -> int:
    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} drifted; run scripts/export_golden.py and review the diff")
            return 1
        print(f"{len(data['cases'])} fixtures up to date")
        return 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(data['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
