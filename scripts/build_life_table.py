"""Build the baseline life table from the 2020 census (NBS) raw table.

Source: 国家统计局《中国人口普查年鉴-2020》表 6-4
"全国分年龄、性别的死亡人口状况 (2019.11.1-2020.10.31)",
https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/A0604.xls

Census-reported deaths are known to be incomplete, so the raw rates give a
life expectancy above the official figure. We scale each sex's mortality by a
single factor so that e0 matches the official 2020 life expectancy published
in 《中国统计年鉴2025》表 2-3 (男 75.37 / 女 80.88). The scale factor is an
estimate (registry id: baseline.census_completeness_scale).

Usage (needs `xlrd`, dev only):
    python scripts/build_life_table.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "census2020_table6-4.xls"
OUT = ROOT / "data" / "cn_life_table_2020.csv"

OFFICIAL_E0 = {"male": 75.37, "female": 80.88}
MAX_AGE = 110  # table is closed with a Gompertz tail fitted on ages 80-94
TAIL_FIT_AGES = range(80, 95)


def read_census() -> dict[str, dict[int, tuple[float, float]]]:
    """Return {sex: {age: (population, deaths)}} for single ages 0..100(+)."""
    sheet = xlrd.open_workbook(str(RAW)).sheet_by_index(0)
    out: dict[str, dict[int, tuple[float, float]]] = {"male": {}, "female": {}}
    for r in range(sheet.nrows):
        row = sheet.row_values(r)
        label = row[0]
        if isinstance(label, float):
            age = int(label)
        elif label == "0":
            age = 0
        elif isinstance(label, str) and label.startswith("100"):
            age = 100
        else:
            continue
        out["male"][age] = (float(row[2]), float(row[5]))
        out["female"][age] = (float(row[3]), float(row[6]))
    for sex, table in out.items():
        missing = [a for a in range(101) if a not in table]
        if missing:
            raise ValueError(f"{sex}: census table missing ages {missing}")
    return out


def gompertz_tail(mx: list[float]) -> tuple[float, float]:
    """Least-squares fit of ln m = a + b*age on TAIL_FIT_AGES."""
    xs = list(TAIL_FIT_AGES)
    ys = [math.log(mx[a]) for a in xs]
    n = len(xs)
    xbar, ybar = sum(xs) / n, sum(ys) / n
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
    return ybar - b * xbar, b


def raw_mx(table: dict[int, tuple[float, float]]) -> list[float]:
    """Single-age central death rates 0..MAX_AGE; ages >= 95 from the Gompertz tail.

    Reported rates flatten above ~95 (age misreporting), so the tail is
    extrapolated rather than read from the census.
    """
    mx = [table[a][1] / table[a][0] for a in range(95)]
    a, b = gompertz_tail(mx)
    mx += [math.exp(a + b * age) for age in range(95, MAX_AGE + 1)]
    return mx


def life_expectancy(mx: list[float], from_age: int = 0) -> float:
    """Period life expectancy from single-age m_x (a_x = 0.5, a_0 = 0.1)."""
    survivors, total = 1.0, 0.0
    for age in range(from_age, len(mx)):
        m = mx[age]
        ax = 0.1 if age == 0 else 0.5
        qx = 1.0 if age == len(mx) - 1 else m / (1 + (1 - ax) * m)
        qx = min(qx, 1.0)
        deaths = survivors * qx
        if age == len(mx) - 1:
            total += survivors / m
        else:
            total += survivors - (1 - ax) * deaths
        survivors -= deaths
    return total


def calibrate_scale(mx: list[float], target_e0: float) -> float:
    lo, hi = 0.5, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if life_expectancy([m * mid for m in mx]) > target_e0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    census = read_census()
    rows = []
    for sex in ("male", "female"):
        mx = raw_mx(census[sex])
        scale = calibrate_scale(mx, OFFICIAL_E0[sex])
        print(
            f"{sex}: raw census e0={life_expectancy(mx):.2f}, "
            f"scale={scale:.4f}, calibrated e0={life_expectancy([m * scale for m in mx]):.2f}, "
            f"e30={life_expectancy([m * scale for m in mx], 30):.2f}"
        )
        for age, m in enumerate(mx):
            pop, deaths = census[sex].get(age, (None, None)) if age <= 100 else (None, None)
            rows.append(
                {
                    "sex": sex,
                    "age": age,
                    "census_population": "" if pop is None else int(pop),
                    "census_deaths": "" if deaths is None else int(deaths),
                    "mx_raw": f"{m:.8f}",
                    "mx": f"{m * scale:.8f}",
                    "source": "census" if age < 95 else "gompertz_tail_80_94",
                }
            )
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
