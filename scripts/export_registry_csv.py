"""Export registry/parameters.json to a flat CSV (one row per parameter).

Nested values are JSON-encoded inside their cell. tests/test_registry.py
fails if the committed CSV is out of date.

Usage:
    python scripts/export_registry_csv.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "registry" / "parameters.json"
CSV_PATH = ROOT / "registry" / "parameters.csv"

COLUMNS = (
    "id",
    "domain",
    "variable",
    "exposure_definition",
    "reference_level",
    "unit",
    "effect_type",
    "estimate_male",
    "estimate_female",
    "range_male",
    "range_female",
    "population",
    "source",
    "source_citation",
    "doi",
    "pmid",
    "source_location",
    "evidence_type",
    "status",
    "estimated",
    "estimate_note",
    "rationale",
    "notes",
    "used_in",
)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_csv() -> str:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    sources = data["sources"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for p in data["parameters"]:
        src = sources[p["source"]]
        rng = p.get("range") or {}
        writer.writerow(
            {
                "id": p["id"],
                "domain": p["domain"],
                "variable": p["variable"],
                "exposure_definition": p["exposure_definition"],
                "reference_level": p["reference_level"],
                "unit": p["unit"],
                "effect_type": p["effect_type"],
                "estimate_male": _cell(p["estimate"].get("male")),
                "estimate_female": _cell(p["estimate"].get("female")),
                "range_male": _cell(rng.get("male")),
                "range_female": _cell(rng.get("female")),
                "population": p["population"],
                "source": p["source"],
                "source_citation": src["citation"],
                "doi": src.get("doi", ""),
                "pmid": src.get("pmid", ""),
                "source_location": p.get("source_location", ""),
                "evidence_type": p["evidence_type"],
                "status": p["status"],
                "estimated": "true" if p["estimated"] else "false",
                "estimate_note": p["estimate_note"],
                "rationale": p.get("rationale", ""),
                "notes": p.get("notes", ""),
                "used_in": _cell(p.get("used_in", [])),
            }
        )
    return buf.getvalue()


def main() -> None:
    CSV_PATH.write_text(render_csv(), encoding="utf-8-sig")
    print(f"wrote {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
