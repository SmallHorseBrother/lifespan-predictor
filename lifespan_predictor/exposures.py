"""User profile, input validation and evidence-range flags."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import SEXES, load_registry

SMOKING = ("never", "former", "current_lt20", "current_ge20")
ALCOHOL = ("less_than_daily", "daily_1_29", "daily_30_59", "daily_ge60")

CKB_BASELINE_AGE = (30.0, 79.0)
BMI_SUPPORTED = (15.0, 45.0)


@dataclass(frozen=True)
class Profile:
    """One person's state at the moment the three futures fork.

    energy_balance_kcal: long-run daily energy surplus (+) or deficit (-)
    relative to weight maintenance in the "current" future. 0 = weight stable.
    """

    sex: str
    age: float
    height_cm: float
    weight_kg: float
    smoking: str = "never"
    alcohol: str = "less_than_daily"
    pa_met_h: float = 16.0
    fruit_g: float = 80.0
    fish_g: float = 20.0
    sleep_h: float = 7.5
    energy_balance_kcal: float = 0.0

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100) ** 2


_LIMITS = {
    "age": (18.0, 100.0),
    "height_cm": (120.0, 220.0),
    "weight_kg": (30.0, 250.0),
    "pa_met_h": (0.0, 150.0),
    "fruit_g": (0.0, 2000.0),
    "fish_g": (0.0, 1000.0),
    "sleep_h": (0.0, 16.0),
    "energy_balance_kcal": (-1500.0, 1500.0),
}


def validate(profile: Profile) -> Profile:
    """Raise ValueError with a readable message if any input is implausible."""
    errors = []
    if profile.sex not in SEXES:
        errors.append(f"sex 必须是 {SEXES} 之一")
    if profile.smoking not in SMOKING:
        errors.append(f"smoking 必须是 {SMOKING} 之一")
    if profile.alcohol not in ALCOHOL:
        errors.append(f"alcohol 必须是 {ALCOHOL} 之一")
    for name, (lo, hi) in _LIMITS.items():
        value = getattr(profile, name)
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            errors.append(f"{name}={value!r} 超出允许范围 [{lo}, {hi}]")
    if errors:
        raise ValueError("输入无效：" + "；".join(errors))
    return profile


def evidence_warnings(
    sex: str, age: float, pa_met_h: float, fruit_g: float, fish_g: float, sleep_h: float, bmi: float
) -> list[dict[str, str]]:
    """Inputs outside the range the source studies observed: {code, field, message}."""
    reg = load_registry()
    out: list[dict[str, str]] = []

    def add(field: str, message: str) -> None:
        out.append({"code": "EXTRAPOLATED", "field": field, "message": message})

    if not CKB_BASELINE_AGE[0] <= age <= CKB_BASELINE_AGE[1]:
        add("person.age_years", f"年龄 {age:g} 不在 CKB 基线年龄 30-79 岁内，风险比按原值外推")
    _, pa_hi = reg.value("pa.observed_range", sex)
    if pa_met_h > pa_hi:
        add("activity.pa_met_h", f"身体活动 {pa_met_h:g} MET-h/d 超过原模型最大值 {pa_hi}")
    if pa_met_h < reg.value("pa.floor", sex):
        add("activity.pa_met_h", f"身体活动 {pa_met_h:g} MET-h/d 低于对数截断值，按截断值计算")
    observed = reg.value("diet.observed_range", sex)
    caps = reg.value("diet.effect_caps", sex)
    for name, value in (("fruit", fruit_g), ("fish", fish_g)):
        lo, hi = observed[name]
        label = "水果" if name == "fruit" else "鱼/海产品"
        if not lo <= value <= hi:
            note = "，超出封顶值部分不再计入" if value > caps[name] else ""
            add(f"diet.{name}_g", f"{label} {value:g} g/d 不在 CKB 观察范围 {lo}-{hi} g/d{note}")
    cut = reg.value("sleep.cutpoints", sex)
    if sleep_h < cut[0] or sleep_h > cut[-1]:
        add("sleep.hours_median", f"睡眠 {sleep_h:g} h 超出主要队列覆盖范围，长期效应为外推")
    if not BMI_SUPPORTED[0] <= bmi <= BMI_SUPPORTED[1]:
        add("person.bmi", f"BMI {bmi:.1f} 超出 {BMI_SUPPORTED[0]}-{BMI_SUPPORTED[1]}，风险为外推")
    return out


def evidence_flags(
    sex: str, age: float, pa_met_h: float, fruit_g: float, fish_g: float, sleep_h: float, bmi: float
) -> list[str]:
    """Human-readable messages of evidence_warnings."""
    return [w["message"] for w in evidence_warnings(sex, age, pa_met_h, fruit_g, fish_g, sleep_h, bmi)]
