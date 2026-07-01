"""Terminal progress helpers for the end-to-end emulator pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import TextIO


_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "cache_initial_hifi": ("CACHE", "Initial training spectrum cache"),
    "cache_comparison_hifi": ("CACHE", "Fixed-budget comparison spectrum cache"),
    "cache_validation_truth": ("CACHE", "Validation truth spectrum cache"),
    "cache_resume_hifi": ("CACHE", "Resume checkpoint training spectrum cache"),
    "module1_camb": ("CAMB", "Module1 high-fidelity spectrum evaluation"),
    "module2_gp_fit": ("GP", "Module2 GP fit (per PCA component model stage)"),
    "module3_objective_prepare": ("M3", "Module3 build posterior-variance objective set"),
    "module3_hull_delaunay": ("M3", "Module3 build shared hull Delaunay"),
    "module3_hull_geometry": ("M3", "Module3 build hull geometry"),
    "module3_domain_geometry": ("M3", "Module3 build domain support geometry"),
    "module3_repr_scoring": ("M3", "Module3 representative-point scoring"),
    "module3_local_search": ("M3", "Module3 simplex search"),
    "module3_refinement": ("M3", "Module3 top-K simplex refinement"),
    "module3_refinement_stage1": ("M3", "Module3 stage-1 top simplex refinement"),
    "module3_refinement_stage2": ("M3", "Module3 stage-2 polish refinement"),
    "module3_finalize": ("M3", "Module3 deduplicate and finalize batch"),
    "validation_camb_truth": ("VALIDATION", "Validation truth evaluation"),
}


@dataclass(slots=True)
class PipelineProgressDisplay:
    """Lightweight progress printer used by Autorunner and CLI entrypoints."""

    stream: TextIO = sys.stdout
    _last_counts: dict[str, tuple[int, int]] = field(default_factory=dict)

    def announce(self, message: str) -> None:
        text = str(message).strip()
        if text:
            print(f"[Pipeline] {text}", file=self.stream, flush=True)

    def step(self, index: int, total: int, title: str) -> None:
        print(
            f"[Step {int(index)}/{int(total)}] {str(title).strip()}",
            file=self.stream,
            flush=True,
        )

    def callback(self, stage: str, current: int, total: int) -> None:
        normalized = str(stage).strip()
        current_pair = (int(current), int(total))
        previous = self._last_counts.get(normalized)
        if previous == current_pair:
            return
        self._last_counts[normalized] = current_pair
        category, label = _STAGE_LABELS.get(
            normalized,
            ("TASK", normalized.replace("_", " ").strip() or "unknown"),
        )
        print(
            f"[{category}] {label}: {int(current)}/{int(total)}",
            file=self.stream,
            flush=True,
        )

    def cache_hit(self, cache_name: str, cache_path: str | Path) -> None:
        print(
            f"[CACHE] Reusing cache {str(cache_name).strip()}: {Path(cache_path).resolve()}",
            file=self.stream,
            flush=True,
        )

    def cache_build(self, cache_name: str, cache_path: str | Path) -> None:
        print(
            f"[CACHE] Building cache {str(cache_name).strip()}: {Path(cache_path).resolve()}",
            file=self.stream,
            flush=True,
        )
