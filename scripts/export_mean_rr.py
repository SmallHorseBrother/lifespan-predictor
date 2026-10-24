"""Export the point-estimate recentering table to data/mean_rr_point.csv.

mean_RR(age, sex) is the reference population's average relative risk at each
single age (registry point estimates, age attenuation applied). A Go port can
embed this table instead of regenerating the synthetic CNHS population; the
deterministic curves in lifespan_predictor.foodlink use exactly these values.

Usage:
    python scripts/export_mean_rr.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifespan_predictor.hazards import point_coefficients  # noqa: E402
from lifespan_predictor.population import mean_rr_by_age, reference_population  # noqa: E402
from lifespan_predictor.registry import load_registry  # noqa: E402

OUT = ROOT / "data" / "mean_rr_point.csv"


def rows() -> list[dict]:
    out = []
    for sex in ("male", "female"):
        table = mean_rr_by_age(reference_population(sex), point_coefficients(sex))[0]
        out += [{"sex": sex, "age": age, "mean_rr": f"{value:.10f}"} for age, value in enumerate(table)]
    return out


def main() -> None:
    with OUT.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# registry_version={load_registry().version}\n")
        writer = csv.DictWriter(f, fieldnames=["sex", "age", "mean_rr"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
