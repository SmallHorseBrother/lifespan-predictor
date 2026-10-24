"""Command line entry: python -m lifespan_predictor --sex male --age 30 ...

Prints a JSON summary; --plot writes survival and weight charts (needs
matplotlib).
"""

from __future__ import annotations

import argparse
import json
import sys

from .exposures import ALCOHOL, SMOKING, Profile
from .foodlink import lifespan_result, one_life
from .simulate import simulate


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lifespan_predictor", description="寿命预测器（模拟估算，不用于诊断）")
    p.add_argument("--sex", choices=("male", "female"), required=True)
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--height", type=float, required=True, help="身高 cm")
    p.add_argument("--weight", type=float, required=True, help="体重 kg")
    p.add_argument("--smoking", choices=SMOKING, default="never")
    p.add_argument("--alcohol", choices=ALCOHOL, default="less_than_daily")
    p.add_argument("--pa", type=float, default=16.0, help="总身体活动 MET-h/天")
    p.add_argument("--fruit", type=float, default=80.0, help="水果 g/天")
    p.add_argument("--fish", type=float, default=20.0, help="鱼/海产品 g/天")
    p.add_argument("--sleep", type=float, default=7.5, help="睡眠 小时/晚")
    p.add_argument("--energy-balance", type=float, default=0.0, help="长期每日能量盈余 kcal（0=体重稳定）")
    p.add_argument("--lives", type=int, default=10_000)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--lag", type=float, default=None, help="行为改变的滞后时间常数（年）")
    p.add_argument("--plot", default=None, help="把生存曲线和体重轨迹画到这个 PNG 文件")
    p.add_argument("--mode", choices=("lifestyle", "diet"), default="lifestyle", help="lifestyle=多因素分叉；diet=只改饮食（食探默认）")
    p.add_argument("--format", choices=("summary", "foodlink", "one-life"), default="summary", help="输出格式")
    p.add_argument("--life-index", type=int, default=0, help="--format one-life 时的人生编号")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    profile = Profile(
        sex=args.sex,
        age=args.age,
        height_cm=args.height,
        weight_kg=args.weight,
        smoking=args.smoking,
        alcohol=args.alcohol,
        pa_met_h=args.pa,
        fruit_g=args.fruit,
        fish_g=args.fish,
        sleep_h=args.sleep,
        energy_balance_kcal=args.energy_balance,
    )
    try:
        if args.format == "foodlink":
            payload = lifespan_result(profile, mode=args.mode, seed=args.seed, n_param_draws=args.draws, lag_tau_years=args.lag)
        elif args.format == "one-life":
            payload = one_life(profile, mode=args.mode, seed=args.seed, life_index=args.life_index, lag_tau_years=args.lag)
        else:
            result = simulate(
                profile, n_lives=args.lives, n_param_draws=args.draws, seed=args.seed, lag_tau_years=args.lag, mode=args.mode
            )
            payload = result.to_dict()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.plot and args.format == "summary":
        from .plots import plot_result

        plot_result(result, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
