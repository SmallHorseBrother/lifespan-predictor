"""Read-only access to registry/parameters.json.

Every number the engine uses comes through this module, so that the Go
backend in FoodLink can read the same file and get the same results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "parameters.json"

SEXES = ("male", "female")
STATUSES = ("论文", "估算")
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Parameter:
    id: str
    status: str
    effect_type: str
    estimate: dict[str, Any]
    range: dict[str, Any] | None
    source: str
    raw: dict[str, Any]

    def value(self, sex: str) -> Any:
        return self.estimate[sex]

    def standard_error(self, sex: str) -> float:
        """SE on the estimate's scale, from a 95% CI (论文) or plausible range (估算)."""
        if self.range is None:
            return 0.0
        lo, hi = self.range[sex]
        return (hi - lo) / (2 * Z95)


@dataclass(frozen=True)
class Registry:
    version: str
    parameters: dict[str, Parameter]
    sources: dict[str, dict[str, Any]]

    def __getitem__(self, parameter_id: str) -> Parameter:
        try:
            return self.parameters[parameter_id]
        except KeyError as exc:
            raise KeyError(f"unknown registry parameter: {parameter_id}") from exc

    def value(self, parameter_id: str, sex: str) -> Any:
        if sex not in SEXES:
            raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
        return self[parameter_id].value(sex)


def _parse(data: dict[str, Any]) -> Registry:
    sources = data["sources"]
    params: dict[str, Parameter] = {}
    for raw in data["parameters"]:
        pid = raw["id"]
        if pid in params:
            raise ValueError(f"duplicate parameter id {pid}")
        if raw["status"] not in STATUSES:
            raise ValueError(f"{pid}: status must be one of {STATUSES}")
        if raw["source"] not in sources:
            raise ValueError(f"{pid}: unknown source {raw['source']}")
        if raw["status"] == "估算" and not raw.get("rationale"):
            raise ValueError(f"{pid}: estimated parameters need a rationale")
        params[pid] = Parameter(
            id=pid,
            status=raw["status"],
            effect_type=raw["effect_type"],
            estimate=raw["estimate"],
            range=raw.get("range"),
            source=raw["source"],
            raw=raw,
        )
    return Registry(version=data["registry_version"], parameters=params, sources=sources)


@lru_cache(maxsize=4)
def load_registry(path: str | None = None) -> Registry:
    target = Path(path) if path else REGISTRY_PATH
    with target.open(encoding="utf-8") as f:
        return _parse(json.load(f))
