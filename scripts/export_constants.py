"""Export every constant the deterministic engine needs to golden/constants.json.

Go and JS ports load this one file instead of re-deriving anything:
point coefficients, the calibrated life table (m_x), the recentering table
(mean_RR), registry values used by scenarios / weight model / warnings /
plan, and the fixed algorithm constants. Replaces food_link's
lifespan-ui/engine-port/export_bundle.py (same "sex" layout, more keys).

Usage:
    python scripts/export_constants.py           # write
    python scripts/export_constants.py --check   # exit 1 if the file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifespan_predictor.exposures import ALCOHOL, SMOKING  # noqa: E402
from lifespan_predictor.foodlink import ENGINE_VERSION, GOLDEN_GAMMA, HISTOGRAM_BIN_YEARS, params_sha256  # noqa: E402
from lifespan_predictor.hazards import point_coefficients  # noqa: E402
from lifespan_predictor.life_table import MAX_AGE, baseline_mx  # noqa: E402
from lifespan_predictor.population import mean_rr_by_age, reference_population  # noqa: E402
from lifespan_predictor.registry import load_registry  # noqa: E402
from lifespan_predictor.weight import DAYS_PER_YEAR, KCAL_TO_MJ  # noqa: E402

OUT = ROOT / "golden" / "constants.json"
REGISTRY_KEYS = (
    "age.effect_attenuation",
    "lag.effect_tau_years",
    "plan.adherence",
    "plan.delayed_start",
    "hall.energy_expenditure",
    "hall.rho_fat",
    "hall.rho_lean",
    "hall.forbes_c",
    "hall.initial_body_fat",
    "scenario.diet_right_targets",
    "scenario.diet_loose_targets",
    "scenario.optimized_targets",
    "scenario.adverse_targets",
    "scenario.adverse_energy_surplus",
    "pa.observed_range",
    "pa.floor",
    "diet.observed_range",
    "diet.effect_caps",
    "sleep.cutpoints",
)


def _coefficients(sex: str) -> dict:
    c = point_coefficients(sex)
    out = {}
    for name in c.__dataclass_fields__:
        value = getattr(c, name)
        out[name] = float(value[0, 0]) if hasattr(value, "shape") else (list(value) if isinstance(value, tuple) else value)
    return out


def build() -> dict:
    reg = load_registry()
    return {
        "engine_version": ENGINE_VERSION,
        "registry_version": reg.version,
        "registry_sha256": params_sha256(),
        "algorithm": {
            "max_age": MAX_AGE,
            "kcal_to_mj": KCAL_TO_MJ,
            "days_per_year": DAYS_PER_YEAR,
            "weight_step_days": 7.0,
            "weight_steps_per_year": 52,
            "weight_schedule_rule": "energy balance read once per step at t = (j + 0.5) * h / 365.25 years, j = global step index",
            "histogram_bin_years": HISTOGRAM_BIN_YEARS,
            "smoking_levels": list(SMOKING),
            "alcohol_levels": list(ALCOHOL),
            "rng": {
                "name": "splitmix64",
                "gamma": hex(GOLDEN_GAMMA),
                "life_uniform": "u = ((splitmix64((seed * gamma + life_index) mod 2^64) >> 11) + 0.5) * 2^-53; threshold = -ln(1 - u)",
            },
        },
        "sex": {
            sex: {
                "mx": [float(x) for x in baseline_mx(sex)],
                "mean_rr": [float(x) for x in mean_rr_by_age(reference_population(sex), point_coefficients(sex))[0]],
                "coef": _coefficients(sex),
                "reg": {key: reg.value(key, sex) for key in REGISTRY_KEYS},
            }
            for sex in ("male", "female")
        },
    }


def render() -> str:
    return json.dumps(build(), ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale; run scripts/export_constants.py")
            return 1
        print("constants up to date")
        return 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
