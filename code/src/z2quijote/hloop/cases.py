"""Case registry loading and validation for lightweight harness runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Case:
    """One reproducible behavior check."""

    id: str
    input: Any
    expected: Any
    grader: Mapping[str, Any] = field(default_factory=lambda: {"type": "exact"})
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    timeout_s: float = 2.0


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("case registry must contain a list field named 'cases'")

    cases: list[Case] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each case must be an object")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each case needs a non-empty string id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)

        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"case {case_id} has non-positive weight")

        timeout_s = float(raw.get("timeout_s", 2.0))
        if timeout_s <= 0:
            raise ValueError(f"case {case_id} has non-positive timeout_s")

        cases.append(
            Case(
                id=case_id,
                input=raw.get("input"),
                expected=raw.get("expected"),
                grader=raw.get("grader", {"type": "exact"}),
                tags=tuple(raw.get("tags", ())),
                weight=weight,
                timeout_s=timeout_s,
            )
        )
    return cases
