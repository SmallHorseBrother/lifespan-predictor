"""Print the calibration anchor table and write docs/anchor_results.md.

Usage:
    python scripts/run_anchors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifespan_predictor.anchors import anchor_report  # noqa: E402
from lifespan_predictor.registry import load_registry  # noqa: E402

OUT = ROOT / "docs" / "anchor_results.md"
SEX_CN = {"male": "男", "female": "女"}
NC_ROWS = (
    ("ideal", "理想情景（四项合计）"),
    ("practical", "现实情景（四项合计）"),
    ("ideal_diet", "理想·饮食单项"),
    ("ideal_pa", "理想·身体活动单项"),
    ("ideal_smoking", "理想·吸烟单项"),
    ("ideal_alcohol", "理想·饮酒单项"),
)


def render() -> str:
    r = anchor_report()
    reg = load_registry()
    nc_ci = reg["anchor.nc2025"].range
    ckb_ci = reg["anchor.ckb2022"].range
    lines = [
        "# 校准锚点检查结果",
        "",
        f"由 `python scripts/run_anchors.py` 生成；参数注册表版本 `{reg.version}`。点估计、无滞后（τ=0），与原论文口径一致。",
        "",
        "## 1. Nat Commun 2025 全人群情景：30 岁预期寿命增加（年）",
        "",
        "| 性别 | 情景 | 本模型 | 论文 | 论文 95% CI | 差 |",
        "|---|---|---:|---:|---|---:|",
    ]
    for sex in ("male", "female"):
        for key, label in NC_ROWS:
            row = r["nc2025"][sex][key]
            ci = nc_ci[sex].get(key)
            ci_txt = f"{ci[0]}–{ci[1]}" if ci else "—"
            diff = row["model"] / row["paper"] - 1
            lines.append(
                f"| {SEX_CN[sex]} | {label} | {row['model']:.2f} | {row['paper']} | {ci_txt} | {diff:+.0%} |"
            )
    lines += [
        "",
        "## 2. Lancet Public Health 2022：5 项低风险 vs 0–1 项",
        "",
        "| 性别 | 本模型 30 岁 LE 差 | 论文 | 论文 95% CI | 差 | 本模型 RR | 论文 HR（男女合并） |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for sex in ("male", "female"):
        row = r["ckb2022"][sex]
        ci = ckb_ci[sex]["le_gain"]
        lines.append(
            f"| {SEX_CN[sex]} | {row['gain']:.2f} | {row['paper_gain']} | {ci[0]}–{ci[1]} | "
            f"{row['gain'] / row['paper_gain'] - 1:+.0%} | {row['rr_all5_vs_0_1']:.3f} | {row['paper_hr']} |"
        )
    lines += [
        "",
        "参照人群中 5 项全满足 / 0–1 项的占比："
        + "；".join(
            f"{SEX_CN[s]} {r['ckb2022'][s]['share_all5']:.1%} / {r['ckb2022'][s]['share_0_1']:.1%}" for s in ("male", "female")
        ),
        "",
        "## 3. 基础生命表",
        "",
        "| 性别 | 校准后 e0 | 官方 2020 e0 | 校准后 e30 | Nat Commun 2025 基准 e30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sex in ("male", "female"):
        row = r["baseline"][sex]
        lines.append(
            f"| {SEX_CN[sex]} | {row['e0']:.2f} | {row['e0_official']} | {row['e30']:.2f} | {row['e30_nc2025_base']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    text = render()
    OUT.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
