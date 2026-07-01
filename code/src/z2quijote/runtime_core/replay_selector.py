"""External module3 selection replay without embedding a search algorithm."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from z2quijote.runtime_core.config import ValidationRuntimeConfig, denormalize_theta_batch, normalize_theta_batch
from z2quijote.runtime_core.data_source import active_theta_bounds
from z2quijote.runtime_core.module3_facade import Module3Selector
from z2quijote.runtime_core.types import Module3ContinuousInput, SelectionResult


@dataclass(slots=True)
class ExternalSelectionReplaySelector(Module3Selector):
    """Load precomputed iteration selections from JSON files.

    Expected file names:
    - ``selection_iteration_01.json``
    - ``selection_iteration_02.json``

    Each JSON file may provide either ``selected_raw_thetas`` or
    ``selected_unit_thetas``. Optional fields:
    - ``selected_source_pc``
    - ``selected_scores``
    - ``metadata``
    """

    selection_dir: Path
    filename_prefix: str = "selection_iteration_"

    def _selection_path(self, iteration_index: int) -> Path:
        return (Path(self.selection_dir).resolve() / f"{self.filename_prefix}{int(iteration_index):02d}.json").resolve()

    def _load_payload(self, iteration_index: int) -> dict[str, object]:
        path = self._selection_path(iteration_index)
        if not path.exists():
            raise FileNotFoundError(
                f"Selection replay file not found for iteration {iteration_index}: {path}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def select_next_batch(
        self,
        config: ValidationRuntimeConfig,
        module3_input: Module3ContinuousInput,
        *,
        progress_callback=None,
    ) -> SelectionResult:
        del progress_callback
        iteration_index = int(module3_input.iteration_index)
        if iteration_index <= 0:
            raise ValueError("module3 continuous input is missing a valid iteration_index for replay.")
        payload = self._load_payload(iteration_index)

        raw = payload.get("selected_raw_thetas")
        unit = payload.get("selected_unit_thetas")
        if raw is None and unit is None:
            raise ValueError(
                f"Replay selection file for iteration {iteration_index} must define "
                "`selected_raw_thetas` or `selected_unit_thetas`."
            )
        if raw is None:
            selected_unit = np.asarray(unit, dtype=np.float64)
            selected_raw = denormalize_theta_batch(selected_unit, active_theta_bounds(config))
        elif unit is None:
            selected_raw = np.asarray(raw, dtype=np.float64)
            selected_unit = normalize_theta_batch(selected_raw, active_theta_bounds(config))
        else:
            selected_raw = np.asarray(raw, dtype=np.float64)
            selected_unit = np.asarray(unit, dtype=np.float64)

        if selected_raw.ndim != 2 or selected_unit.ndim != 2:
            raise ValueError("Replay selector expects 2D theta arrays.")
        if selected_raw.shape[0] != selected_unit.shape[0]:
            raise ValueError("selected_raw_thetas and selected_unit_thetas row counts must match.")

        batch_size = selected_raw.shape[0]
        source_pc = payload.get("selected_source_pc")
        if source_pc is None:
            source_pc_arr = np.full((batch_size,), -1, dtype=np.int64)
        else:
            source_pc_arr = np.asarray(source_pc, dtype=np.int64).reshape(-1)
            if source_pc_arr.shape[0] != batch_size:
                raise ValueError("selected_source_pc length must match selected theta count.")
        scores = payload.get("selected_scores")
        if scores is None:
            scores_arr = np.zeros((batch_size,), dtype=np.float64)
        else:
            scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
            if scores_arr.shape[0] != batch_size:
                raise ValueError("selected_scores length must match selected theta count.")

        return SelectionResult(
            selected_raw_thetas=selected_raw.astype(np.float64),
            selected_unit_thetas=selected_unit.astype(np.float64),
            selected_source_pc=source_pc_arr,
            selected_scores=scores_arr,
            metadata=dict(payload.get("metadata", {})),
        )
