"""Feedback-loop controller for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from z2quijote.hloop.cases import Case
from z2quijote.hloop.runner import Harness, RunResult, Target


@dataclass(frozen=True)
class Candidate:
    name: str
    config: Mapping[str, Any]
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class LoopConfig:
    cost_weight: float = 0.01
    latency_weight: float = 0.001
    risk_weight: float = 0.05


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    results: list[RunResult]
    summary: Mapping[str, Any]
    utility: float


@dataclass(frozen=True)
class LoopReport:
    selected_candidate: str
    environment: Mapping[str, Any]
    loop_config: LoopConfig
    evaluations: list[CandidateEvaluation]


class LoopController:
    """Evaluate candidates and select the best next state by utility."""

    def __init__(self, harness: Harness | None = None, config: LoopConfig | None = None):
        self.harness = harness or Harness()
        self.config = config or LoopConfig()

    def run(
        self,
        cases: list[Case],
        candidates: list[Candidate],
        target: Target,
    ) -> LoopReport:
        if not candidates:
            raise ValueError("at least one candidate is required")

        from z2quijote.hloop.report import summarize_results

        evaluations: list[CandidateEvaluation] = []
        for candidate in candidates:
            results = self.harness.run(cases, candidate.name, target, candidate.config)
            summary = summarize_results(results)
            utility = self._utility(candidate, summary)
            evaluations.append(CandidateEvaluation(candidate, results, summary, utility))

        selected = max(evaluations, key=lambda item: item.utility)
        return LoopReport(
            selected_candidate=selected.candidate.name,
            environment=self.harness.environment(),
            loop_config=self.config,
            evaluations=evaluations,
        )

    def _utility(self, candidate: Candidate, summary: Mapping[str, Any]) -> float:
        return (
            float(summary["weighted_score"])
            - self.config.cost_weight * candidate.cost
            - self.config.latency_weight * (float(summary["avg_latency_ms"]) / 1000.0)
            - self.config.risk_weight * candidate.risk
        )
