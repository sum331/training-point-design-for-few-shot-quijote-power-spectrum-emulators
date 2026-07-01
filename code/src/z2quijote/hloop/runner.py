"""Harness runner with traces, errors, and timing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
import platform
import random
import sys
import time
from typing import Any, Callable, Mapping

from z2quijote.hloop.cases import Case
from z2quijote.hloop.graders import grade_output


Target = Callable[[Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class RunResult:
    case_id: str
    candidate: str
    passed: bool
    score: float
    weight: float
    latency_ms: float
    output: Any = None
    expected: Any = None
    error: str | None = None
    tags: tuple[str, ...] = ()
    grading_message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


class Harness:
    """Run candidates against cases and return structured results."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def environment(self) -> dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seed": self.seed,
        }

    def run_case(
        self,
        case: Case,
        candidate_name: str,
        target: Target,
        config: Mapping[str, Any],
    ) -> RunResult:
        random.seed(self.seed)
        start = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(target, case.input, dict(config))
        try:
            output = future.result(timeout=case.timeout_s)
            latency_ms = (time.perf_counter() - start) * 1000
            grade = grade_output(output, case.expected, case.grader)
            executor.shutdown(wait=True)
            return RunResult(
                case_id=case.id,
                candidate=candidate_name,
                passed=grade.passed,
                score=grade.score,
                weight=case.weight,
                latency_ms=latency_ms,
                output=output,
                expected=case.expected,
                tags=case.tags,
                grading_message=grade.message,
                details=grade.details,
            )
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            latency_ms = (time.perf_counter() - start) * 1000
            return RunResult(
                case_id=case.id,
                candidate=candidate_name,
                passed=False,
                score=0.0,
                weight=case.weight,
                latency_ms=latency_ms,
                expected=case.expected,
                error=f"timeout after {case.timeout_s:.3f}s",
                tags=case.tags,
                grading_message="timeout",
            )
        except Exception as exc:  # noqa: BLE001 - harness must capture target failures.
            executor.shutdown(wait=True)
            latency_ms = (time.perf_counter() - start) * 1000
            return RunResult(
                case_id=case.id,
                candidate=candidate_name,
                passed=False,
                score=0.0,
                weight=case.weight,
                latency_ms=latency_ms,
                expected=case.expected,
                error=f"{type(exc).__name__}: {exc}",
                tags=case.tags,
                grading_message="target error",
            )

    def run(
        self,
        cases: list[Case],
        candidate_name: str,
        target: Target,
        config: Mapping[str, Any],
    ) -> list[RunResult]:
        return [self.run_case(case, candidate_name, target, config) for case in cases]
