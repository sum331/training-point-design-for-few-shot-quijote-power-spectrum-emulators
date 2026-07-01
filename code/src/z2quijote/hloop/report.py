"""Report helpers for harness and loop runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from z2quijote.hloop.loop import CandidateEvaluation, LoopReport
from z2quijote.hloop.runner import RunResult


def summarize_results(results: Iterable[RunResult]) -> dict[str, Any]:
    rows = list(results)
    total_weight = sum(row.weight for row in rows) or 1.0
    weighted_score = sum(row.score * row.weight for row in rows) / total_weight
    pass_count = sum(1 for row in rows if row.passed)
    latencies = [row.latency_ms for row in rows]
    failures_by_tag: Counter[str] = Counter()
    for row in rows:
        if not row.passed:
            failures_by_tag.update(row.tags or ("untagged",))

    return {
        "case_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(rows) if rows else 0.0,
        "weighted_score": weighted_score,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "failures_by_tag": dict(failures_by_tag),
    }


def write_json_report(report: LoopReport, path: str | Path) -> None:
    payload = report_to_dict(report)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(report: LoopReport, path: str | Path) -> None:
    lines = [
        "# Harness Loop Report",
        "",
        f"- selected candidate: `{report.selected_candidate}`",
        f"- seed: `{report.environment.get('seed')}`",
        f"- python: `{report.environment.get('python')}`",
        "",
        "## Candidate Scores",
        "",
        "| candidate | utility | weighted score | pass rate | avg latency ms | failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation in sorted(report.evaluations, key=lambda item: item.utility, reverse=True):
        summary = evaluation.summary
        failures = summary["case_count"] - summary["pass_count"]
        lines.append(
            "| {name} | {utility:.4f} | {score:.4f} | {pass_rate:.2%} | {latency:.2f} | {failures} |".format(
                name=evaluation.candidate.name,
                utility=evaluation.utility,
                score=summary["weighted_score"],
                pass_rate=summary["pass_rate"],
                latency=summary["avg_latency_ms"],
                failures=failures,
            )
        )

    lines.extend(["", "## Failures", ""])
    failures_by_candidate: dict[str, list[RunResult]] = defaultdict(list)
    for evaluation in report.evaluations:
        for result in evaluation.results:
            if not result.passed:
                failures_by_candidate[evaluation.candidate.name].append(result)

    if not any(failures_by_candidate.values()):
        lines.append("No failures.")
    else:
        for candidate, failures in failures_by_candidate.items():
            lines.append(f"### {candidate}")
            for failure in failures:
                reason = failure.error or failure.grading_message
                lines.append(f"- `{failure.case_id}`: {reason}; output={failure.output!r}; expected={failure.expected!r}")
            lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_to_dict(report: LoopReport) -> dict[str, Any]:
    return {
        "selected_candidate": report.selected_candidate,
        "environment": dict(report.environment),
        "loop_config": asdict(report.loop_config),
        "evaluations": [_evaluation_to_dict(evaluation) for evaluation in report.evaluations],
    }


def _evaluation_to_dict(evaluation: CandidateEvaluation) -> dict[str, Any]:
    return {
        "candidate": asdict(evaluation.candidate),
        "utility": evaluation.utility,
        "summary": dict(evaluation.summary),
        "results": [result.to_dict() for result in evaluation.results],
    }
