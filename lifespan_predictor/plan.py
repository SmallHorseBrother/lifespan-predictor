"""How a future is carried out: when the change starts and how well it sticks.

change_start_age (从某个年龄才开始改): before this age the branch lives with
today's habits (the reference branch's exposures and energy balance); from
this age on it switches to its own targets. Lag counts from this age, and the
weight model carries on from the body it has reached by then.

adherence (坚持程度): a(s) in [0, 1], s = years since the change started.
- continuous exposures (activity, fruit, fish, sleep, energy balance) are
  set to  current + a(s) * (target - current);
- categorical exposures (smoking, alcohol) interpolate their log HR:
  old + a(s) * (new - old).
A constant a, or a decaying one  a(s) = end + (start - end) * 0.5 ** (s / half_life).

Both are modelling assumptions (registry plan.*, status 估算). The defaults
(start now, adherence 1) reproduce the plain engine bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .life_table import MAX_AGE


@dataclass(frozen=True)
class Adherence:
    """Constant when end is None, otherwise decays from start to end."""

    start: float = 1.0
    end: float | None = None
    half_life_years: float | None = None

    @property
    def is_full(self) -> bool:
        return self.end is None and self.start == 1.0

    def level(self, years_since_start: np.ndarray) -> np.ndarray:
        s = np.asarray(years_since_start, dtype=float)
        if self.end is None:
            return np.full_like(s, self.start)
        return self.end + (self.start - self.end) * 0.5 ** (s / self.half_life_years)

    def validate(self) -> "Adherence":
        values = [self.start] + ([self.end] if self.end is not None else [])
        if any(not 0.0 <= v <= 1.0 for v in values):
            raise ValueError("adherence 必须在 0 到 1 之间")
        if self.end is not None and not (self.half_life_years and self.half_life_years > 0):
            raise ValueError("衰减型 adherence 需要正的 half_life_years")
        return self

    def to_dict(self) -> dict[str, Any]:
        if self.end is None:
            return {"type": "constant", "level": self.start}
        return {"type": "decay", "start": self.start, "end": self.end, "half_life_years": self.half_life_years}

    @classmethod
    def from_spec(cls, spec: "float | dict | Adherence | None") -> "Adherence":
        """Accept 0.7, {"type": "constant", "level": 0.7} or
        {"type": "decay", "start": 0.9, "end": 0.5, "half_life_years": 2}."""
        if spec is None:
            return cls()
        if isinstance(spec, Adherence):
            return spec.validate()
        if isinstance(spec, (int, float)):
            return cls(start=float(spec)).validate()
        kind = spec.get("type", "constant")
        if kind == "constant":
            return cls(start=float(spec["level"])).validate()
        if kind == "decay":
            return cls(
                start=float(spec["start"]), end=float(spec["end"]), half_life_years=float(spec["half_life_years"])
            ).validate()
        raise ValueError(f"未知的 adherence 类型 {kind!r}")


@dataclass(frozen=True)
class ChangePlan:
    change_start_age: float | None = None  # None = the profile's age (start now)
    adherence: Adherence = Adherence()

    def offset_years(self, age: float) -> float:
        """Years from the profile's age to the start of the change."""
        return 0.0 if self.change_start_age is None else self.change_start_age - age

    def is_default(self, age: float) -> bool:
        return self.offset_years(age) == 0.0 and self.adherence.is_full

    def validate(self, age: float) -> "ChangePlan":
        if self.change_start_age is not None and not age <= self.change_start_age <= MAX_AGE:
            raise ValueError(f"change_start_age 必须在当前年龄 {age:g} 到 {MAX_AGE} 岁之间")
        self.adherence.validate()
        return self

    def to_dict(self, age: float) -> dict[str, Any]:
        start = age if self.change_start_age is None else self.change_start_age
        return {"change_start_age": float(start), "adherence": self.adherence.to_dict()}

    def year_weights(self, age: float, n_years: int) -> tuple[np.ndarray, np.ndarray]:
        """For yearly intervals: years since start at each midpoint (s_k, may be
        negative before the start) and the adherence level there (0 before)."""
        s = np.arange(n_years) + 0.5 - self.offset_years(age)
        level = np.where(s >= 0, self.adherence.level(np.maximum(s, 0.0)), 0.0)
        return s, level

    def energy_schedule(self, age: float, base_kcal: float, target_kcal: float):
        """Daily energy balance (kcal) as a function of years since the profile's age."""
        offset = self.offset_years(age)

        def delta(t_years: float) -> float:
            if t_years < offset:
                return base_kcal
            a = float(self.adherence.level(np.array(t_years - offset)))
            return base_kcal + a * (target_kcal - base_kcal)

        return delta


DEFAULT_PLAN = ChangePlan()


def make_plan(change_start_age: float | None = None, adherence: "float | dict | Adherence | None" = None) -> ChangePlan:
    return ChangePlan(change_start_age=change_start_age, adherence=Adherence.from_spec(adherence))
