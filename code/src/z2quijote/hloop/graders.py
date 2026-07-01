"""Grading functions for harness outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class Grade:
    passed: bool
    score: float
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


def grade_output(output: Any, expected: Any, spec: Mapping[str, Any]) -> Grade:
    grader_type = spec.get("type", "exact")

    if grader_type == "exact":
        passed = output == expected
        return Grade(passed, 1.0 if passed else 0.0, "exact match" if passed else "exact mismatch")

    if grader_type == "contains":
        passed = str(expected) in str(output)
        return Grade(passed, 1.0 if passed else 0.0, "substring found" if passed else "substring missing")

    if grader_type == "regex":
        pattern = str(expected)
        passed = re.search(pattern, str(output)) is not None
        return Grade(passed, 1.0 if passed else 0.0, "regex matched" if passed else "regex did not match")

    if grader_type == "numeric_abs_error":
        tolerance = float(spec.get("tolerance", 0.0))
        try:
            error = abs(float(output) - float(expected))
        except (TypeError, ValueError):
            return Grade(False, 0.0, "numeric conversion failed")
        passed = error <= tolerance
        score = 1.0 if passed else max(0.0, 1.0 - error / max(tolerance, 1e-12))
        if not math.isfinite(score):
            score = 0.0
        return Grade(passed, score, "within tolerance" if passed else "outside tolerance", {"abs_error": error})

    raise ValueError(f"unknown grader type: {grader_type}")
