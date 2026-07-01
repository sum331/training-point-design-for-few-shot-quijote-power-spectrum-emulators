"""Active-learning orchestrator for the cosmology emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np

from z2quijote.runtime_core.camb_data_provider import CAMBDataProvider
from z2quijote.runtime_core.cache_manager import (
    SpectrumBank,
    get_or_create_initial_training_bank,
    get_or_create_spectrum_bank,
    get_or_create_validation_bank,
)
from z2quijote.runtime_core.config import ValidationRuntimeConfig, normalize_theta_batch
from z2quijote.runtime_core.data_source import active_theta_names, resolve_data_source
from z2quijote.runtime_core.dynamic_preprocessing import (
    allocate_band_components,
    compute_band_posterior_variance_scores,
    compute_band_relative_errors,
    compute_proxy_band_scores_from_sensitivity,
    merge_band_weights_to_grid_fractions,
    update_band_weights_from_errors,
)
from z2quijote.runtime_core.evaluation.active_learning_validation import evaluate_emulator_on_validation_set
from z2quijote.runtime_core.evaluation.gp_baseline import run_standard_sobol_gp_baseline
from z2quijote.runtime_core.module1_facade import build_dataset_from_spectrum_bank, build_initial_dataset, extend_dataset
from z2quijote.runtime_core.module2_facade import (
    build_logdiff_projected_component_weights,
    build_continuous_posterior_state,
    build_iteration_pca_band_diagnostics,
    evaluate_continuous_variance,
    fit_emulator,
    predict_spectra,
    summarize_lengthscale_upper_hits,
)
from z2quijote.runtime_core.module3_facade import InterfaceOnlySelector, Module3Selector, select_next_batch
from z2quijote.runtime_core.representation import PCA_BAND_LABELS, resolve_target_transform_from_metadata
from z2quijote.runtime_core.run_artifacts import ensure_run_artifact_layout, run_process_path, run_results_path, run_results_subdir
from z2quijote.runtime_core.sampling import generate_unit_sobol_samples
from z2quijote.runtime_core.types import EmulatorState, IterationRecord, Module1Dataset, Module3ContinuousInput, ValidationArtifacts

ExtensionHook = Callable[["ValidationContext"], dict[str, Any] | None]
ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class ValidationContext:
    config: ValidationRuntimeConfig
    run_dir: Path
    results_dir: Path
    process_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    extension_outputs: dict[str, Any] = field(default_factory=dict)
    initial_raw_thetas: np.ndarray | None = None
    dataset: Module1Dataset | None = None
    emulator: EmulatorState | None = None
    iteration_records: list[IterationRecord] = field(default_factory=list)
    validation: ValidationArtifacts | None = None


class ValidationOrchestrator:
    """Coordinate module1/module2 training, module3 hand-off, and validation."""

    def __init__(
        self,
        config: ValidationRuntimeConfig,
        *,
        camb_data_provider: CAMBDataProvider | None = None,
        batch_selector: Module3Selector | None = None,
        extension_hooks: dict[str, ExtensionHook] | None = None,
        progress_callback: ProgressCallback | None = None,
        initial_dataset: Module1Dataset | None = None,
        validation_bank: SpectrumBank | None = None,
        use_initial_training_cache: bool = True,
        force_rebuild_cache: bool = False,
    ) -> None:
        self.config = config
        self.camb_data_provider = camb_data_provider or CAMBDataProvider(config=config)
        self.batch_selector = batch_selector
        self.progress_callback = progress_callback
        self.extension_hooks: dict[str, ExtensionHook] = dict(extension_hooks or {})
        self.initial_dataset = initial_dataset
        self.validation_bank = validation_bank
        self.use_initial_training_cache = bool(use_initial_training_cache)
        self.force_rebuild_cache = bool(force_rebuild_cache)

    def register_extension(self, name: str, hook: ExtensionHook) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Extension name must be non-empty.")
        self.extension_hooks[normalized] = hook

    def create_run_dir(self, *, timestamp: datetime | None = None) -> Path:
        stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.config.project_root) / self.config.reports_dir / "runs" / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir.resolve()

    def _emit_progress(self, stage: str, current: int, total: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, current, total)

    @staticmethod
    def _parse_module3_iteration_index(path: Path) -> int | None:
        stem = path.stem
        marker = "module3_continuous_input_iteration_"
        if not stem.startswith(marker):
            return None
        suffix = stem[len(marker) :]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _discover_module3_input_artifacts(
        self,
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        process_dir = run_process_path(run_dir, create=True)
        by_iteration: dict[int, dict[str, Any]] = {}
        for path in process_dir.glob("module3_continuous_input_iteration_*.pkl"):
            iteration_index = self._parse_module3_iteration_index(path)
            if iteration_index is None:
                continue
            entry = by_iteration.setdefault(iteration_index, {"iteration_index": int(iteration_index)})
            entry["state_path"] = str(path.resolve())
        for path in process_dir.glob("module3_continuous_input_iteration_*.json"):
            iteration_index = self._parse_module3_iteration_index(path)
            if iteration_index is None:
                continue
            entry = by_iteration.setdefault(iteration_index, {"iteration_index": int(iteration_index)})
            entry["summary_path"] = str(path.resolve())
        return [by_iteration[idx] for idx in sorted(by_iteration)]

    @staticmethod
    def _load_module3_input(path: Path) -> Module3ContinuousInput:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Module3ContinuousInput):
            raise TypeError(f"Expected Module3ContinuousInput, got {type(payload)!r} from {path}.")
        return payload

    @staticmethod
    def _load_module3_summary(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _iteration_record_to_payload(item: IterationRecord) -> dict[str, Any]:
        return {
            "iteration_index": int(item.iteration_index),
            "train_size_before": int(item.train_size_before),
            "train_size_after": int(item.train_size_after),
            "selected_raw_thetas": np.asarray(
                item.selected_raw_thetas,
                dtype=np.float64,
            ).tolist(),
            "selected_unit_thetas": np.asarray(
                item.selected_unit_thetas,
                dtype=np.float64,
            ).tolist(),
            "selected_source_pc": np.asarray(item.selected_source_pc, dtype=np.int64).tolist(),
            "selected_scores": np.asarray(item.selected_scores, dtype=np.float64).tolist(),
            "metadata": dict(item.metadata),
        }

    @staticmethod
    def _payload_to_iteration_record(payload: dict[str, Any]) -> IterationRecord:
        return IterationRecord(
            iteration_index=int(payload["iteration_index"]),
            train_size_before=int(payload["train_size_before"]),
            train_size_after=int(payload["train_size_after"]),
            selected_raw_thetas=np.asarray(
                payload.get("selected_raw_thetas", []),
                dtype=np.float64,
            ),
            selected_unit_thetas=np.asarray(
                payload.get("selected_unit_thetas", []),
                dtype=np.float64,
            ),
            selected_source_pc=np.asarray(
                payload.get("selected_source_pc", []),
                dtype=np.int64,
            ),
            selected_scores=np.asarray(
                payload.get("selected_scores", []),
                dtype=np.float64,
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _reconstruct_training_raw_from_records(
        initial_raw_thetas: np.ndarray,
        records: list[IterationRecord],
    ) -> np.ndarray:
        rows: list[np.ndarray] = [np.asarray(initial_raw_thetas, dtype=np.float64)]
        for record in sorted(records, key=lambda item: int(item.iteration_index)):
            selected = np.asarray(record.selected_raw_thetas, dtype=np.float64)
            if selected.ndim == 1 and selected.size > 0:
                selected = selected.reshape(1, -1)
            if selected.ndim == 2 and selected.shape[0] > 0:
                rows.append(selected)
        return np.vstack(rows).astype(np.float64)

    def _write_iteration_record(
        self,
        context: ValidationContext,
        item: IterationRecord,
    ) -> Path:
        path = run_process_path(
            context.run_dir,
            f"iteration_record_iteration_{int(item.iteration_index):02d}.json",
            create=True,
        )
        path.write_text(
            json.dumps(
                self._iteration_record_to_payload(item),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def _load_saved_iteration_records(
        self,
        run_dir: Path,
    ) -> list[IterationRecord]:
        process_dir = run_process_path(run_dir, create=True)
        records_by_iteration: dict[int, IterationRecord] = {}

        aggregated_path = process_dir / "iteration_history.json"
        if aggregated_path.exists():
            payload = json.loads(aggregated_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    record = self._payload_to_iteration_record(dict(item))
                    records_by_iteration[int(record.iteration_index)] = record

        per_iteration_paths = sorted(
            process_dir.glob("iteration_record_iteration_*.json")
        )
        for path in per_iteration_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = self._payload_to_iteration_record(dict(payload))
            records_by_iteration[int(record.iteration_index)] = record

        return [
            records_by_iteration[idx]
            for idx in sorted(records_by_iteration)
        ]

    def _write_process_state(self, context: ValidationContext) -> Path:
        dataset = context.dataset
        payload = dict(context.metadata)
        payload.update(
            {
                "run_dir": str(context.run_dir),
                "results_dir": str(context.results_dir),
                "process_dir": str(context.process_dir),
                "completed_iterations": int(len(context.iteration_records)),
                "current_train_size": (
                    int(dataset.raw_thetas.shape[0]) if dataset is not None else 0
                ),
                "latest_iteration_record": (
                    int(context.iteration_records[-1].iteration_index)
                    if context.iteration_records
                    else None
                ),
            }
        )
        path = run_process_path(context.run_dir, "resume_state.json", create=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context.metadata["resume_state_path"] = str(path)
        return path

    def _flush_process_progress(self, context: ValidationContext) -> None:
        self._write_iteration_log(context)
        self._write_training_point_summary(context)
        hifi_bank_path = self._write_hifi_bank(context)
        if hifi_bank_path is not None:
            context.metadata["hifi_bank_path"] = str(hifi_bank_path)
        self._write_process_state(context)

    def _save_iteration_progress(
        self,
        context: ValidationContext,
        record: IterationRecord,
    ) -> None:
        self._write_iteration_record(context, record)
        self._flush_process_progress(context)

    @staticmethod
    def _infer_selected_raw_thetas(
        previous_raw_thetas: np.ndarray,
        next_raw_thetas: np.ndarray,
    ) -> np.ndarray:
        previous = np.asarray(previous_raw_thetas, dtype=np.float64)
        current = np.asarray(next_raw_thetas, dtype=np.float64)
        if previous.ndim != 2 or current.ndim != 2:
            raise ValueError("Expected 2D raw theta arrays when reconstructing selections.")
        if current.shape[0] <= previous.shape[0]:
            raise ValueError(
                "Next iteration snapshot must contain a strictly larger training set "
                f"than the previous one, got {previous.shape[0]} -> {current.shape[0]}."
            )
        prefix_rows = previous.shape[0]
        if np.allclose(current[:prefix_rows], previous, atol=1.0e-10, rtol=1.0e-10):
            return current[prefix_rows:].astype(np.float64)

        selected_rows: list[np.ndarray] = []
        previous_keys = {
            tuple(np.round(row, decimals=12).tolist())
            for row in previous
        }
        for row in current:
            key = tuple(np.round(row, decimals=12).tolist())
            if key not in previous_keys:
                selected_rows.append(np.asarray(row, dtype=np.float64))
        if not selected_rows:
            raise ValueError("Could not infer newly selected raw thetas from adjacent checkpoints.")
        return np.vstack(selected_rows).astype(np.float64)

    def _reconstruct_completed_iteration_records(
        self,
        *,
        checkpoint_artifacts: list[dict[str, Any]],
    ) -> tuple[np.ndarray, list[IterationRecord]]:
        if not checkpoint_artifacts:
            raise ValueError("No module3 checkpoint artifacts were found to reconstruct the run state.")

        summaries: dict[int, dict[str, Any]] = {}
        inputs: dict[int, Module3ContinuousInput] = {}
        for artifact in checkpoint_artifacts:
            iteration_index = int(artifact["iteration_index"])
            summary_path = artifact.get("summary_path")
            state_path = artifact.get("state_path")
            if summary_path is None or state_path is None:
                raise FileNotFoundError(
                    "Checkpoint recovery requires both summary and pickle artifacts for each iteration; "
                    f"iteration {iteration_index} is incomplete."
                )
            summaries[iteration_index] = self._load_module3_summary(Path(summary_path))
            inputs[iteration_index] = self._load_module3_input(Path(state_path))

        ordered_iterations = sorted(summaries)
        initial_iteration = ordered_iterations[0]
        initial_raw_thetas = np.asarray(
            summaries[initial_iteration]["train_raw_thetas"],
            dtype=np.float64,
        )

        records: list[IterationRecord] = []
        for previous_iteration, next_iteration in zip(ordered_iterations[:-1], ordered_iterations[1:]):
            previous_summary = summaries[previous_iteration]
            next_summary = summaries[next_iteration]
            previous_raw = np.asarray(previous_summary["train_raw_thetas"], dtype=np.float64)
            next_raw = np.asarray(next_summary["train_raw_thetas"], dtype=np.float64)
            selected_raw = self._infer_selected_raw_thetas(previous_raw, next_raw)
            theta_bounds = np.asarray(
                inputs[previous_iteration].continuous_state.theta_bounds,
                dtype=np.float64,
            )
            selected_unit = normalize_theta_batch(selected_raw, theta_bounds)

            artifact = checkpoint_artifacts[ordered_iterations.index(previous_iteration)]
            records.append(
                IterationRecord(
                    iteration_index=int(previous_iteration),
                    train_size_before=int(previous_raw.shape[0]),
                    train_size_after=int(next_raw.shape[0]),
                    selected_raw_thetas=selected_raw,
                    selected_unit_thetas=selected_unit,
                    selected_source_pc=np.full(
                        (selected_raw.shape[0],),
                        -1,
                        dtype=np.int64,
                    ),
                    selected_scores=np.full(
                        (selected_raw.shape[0],),
                        np.nan,
                        dtype=np.float64,
                    ),
                    metadata={
                        "resume_reconstructed": True,
                        "resume_source": "adjacent_module3_snapshots",
                        "selection_metadata_missing": True,
                        "module3_input_artifact": str(artifact["state_path"]),
                        "module3_summary_artifact": str(artifact["summary_path"]),
                    },
                )
            )
        return initial_raw_thetas.astype(np.float64), records

    def _load_process_hifi_bank_dataset(
        self,
        *,
        run_dir: Path,
        raw_thetas: np.ndarray,
    ) -> Module1Dataset | None:
        process_bank_path = run_process_path(run_dir, "hifi_bank.npz", create=True)
        if not process_bank_path.exists():
            return None
        with np.load(process_bank_path, allow_pickle=False) as npz:
            bank_raw = np.asarray(npz["train_thetas"], dtype=np.float64)
            if bank_raw.ndim != 2 or bank_raw.shape[0] < raw_thetas.shape[0]:
                return None
            prefix_size = int(raw_thetas.shape[0])
            if not np.allclose(
                bank_raw[:prefix_size],
                np.asarray(raw_thetas, dtype=np.float64),
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                return None
            linear_batch = (
                np.asarray(npz["train_linear_pk"], dtype=np.float64)
                if "train_linear_pk" in npz.files
                else None
            )
            dataset = build_dataset_from_spectrum_bank(
                self.config,
                bank_raw[:prefix_size],
                np.asarray(npz["train_k_bins"], dtype=np.float64),
                np.asarray(npz["train_nonlin_pk"], dtype=np.float64)[:prefix_size],
                p_linear_batch=(
                    None if linear_batch is None else linear_batch[:prefix_size]
                ),
            )
        dataset.metadata.update(
            {
                "source": "process_hifi_bank",
                "cache_path": str(process_bank_path),
                "cache_status": "hit",
            }
        )
        return dataset

    def _rebuild_dataset_from_checkpoint(
        self,
        *,
        run_dir: Path,
        raw_thetas: np.ndarray,
        iteration_index: int,
    ) -> Module1Dataset:
        saved_dataset = self._load_process_hifi_bank_dataset(
            run_dir=run_dir,
            raw_thetas=raw_thetas,
        )
        if saved_dataset is not None:
            saved_dataset.metadata.setdefault("resume_iteration_index", int(iteration_index))
            return saved_dataset
        cache_name = f"resume_{run_dir.name}_iter_{int(iteration_index):02d}_training"
        bank = get_or_create_spectrum_bank(
            self.config,
            self.camb_data_provider,
            cache_name=cache_name,
            raw_thetas=np.asarray(raw_thetas, dtype=np.float64),
            asset_version=f"active_learning_resume_iter_{int(iteration_index):02d}",
            stage="cache_resume_hifi",
            progress_callback=self._emit_progress,
            force_rebuild=self.force_rebuild_cache,
        )
        dataset = build_dataset_from_spectrum_bank(
            self.config,
            bank.raw_thetas,
            bank.k_bins,
            bank.p_nonlin_batch,
            p_linear_batch=bank.p_linear_batch,
        )
        dataset.metadata.update(
            {
                "source": "resume_training_cache",
                "cache_name": str(bank.name),
                "cache_path": str(bank.npz_path),
                "cache_status": str(bank.metadata.get("cache_status", "unknown")),
            }
        )
        return dataset

    def _serialize_module3_input(
        self,
        module3_input: Module3ContinuousInput,
        *,
        context: ValidationContext,
    ) -> tuple[Path, Path]:
        iteration_index = int(module3_input.iteration_index)
        state_path = run_process_path(
            context.run_dir,
            f"module3_continuous_input_iteration_{iteration_index:02d}.pkl",
            create=True,
        )
        with state_path.open("wb") as handle:
            pickle.dump(module3_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

        continuous_state = module3_input.continuous_state
        summary_path = run_process_path(
            context.run_dir,
            f"module3_continuous_input_iteration_{iteration_index:02d}.json",
            create=True,
        )
        summary_path.write_text(
            json.dumps(
                {
                    "iteration_index": iteration_index,
                    "batch_size": int(self.config.sampling.batch_size),
                    "train_size": int(continuous_state.train_unit_thetas.shape[0]),
                    "theta_dim": int(continuous_state.train_unit_thetas.shape[1]),
                    "pc_dim": int(len(continuous_state.gp_models)),
                    "theta_bounds": np.asarray(
                        continuous_state.theta_bounds,
                        dtype=np.float64,
                    ).tolist(),
                    "train_raw_thetas": np.asarray(
                        continuous_state.train_raw_thetas,
                        dtype=np.float64,
                    ).tolist(),
                    "train_unit_thetas": np.asarray(
                        continuous_state.train_unit_thetas,
                        dtype=np.float64,
                    ).tolist(),
                    "pc_lengthscales": np.asarray(
                        continuous_state.pc_lengthscales,
                        dtype=np.float64,
                    ).tolist(),
                    "pc_signal_variances": np.asarray(
                        continuous_state.pc_signal_variances,
                        dtype=np.float64,
                    ).tolist(),
                    "kernel_descriptions": list(continuous_state.kernel_descriptions),
                    "continuous_state_metadata": dict(continuous_state.metadata),
                    "metadata": dict(module3_input.metadata),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return state_path, summary_path

    def _write_iteration_log(self, context: ValidationContext) -> Path:
        payload = [
            self._iteration_record_to_payload(item)
            for item in context.iteration_records
        ]
        path = run_process_path(context.run_dir, "iteration_history.json", create=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_training_point_summary(self, context: ValidationContext) -> Path:
        initial_raw = (
            np.asarray(context.initial_raw_thetas, dtype=np.float64)
            if context.initial_raw_thetas is not None
            else np.empty((0, len(active_theta_names(self.config))), dtype=np.float64)
        )
        final_raw = (
            np.asarray(context.dataset.raw_thetas, dtype=np.float64)
            if context.dataset is not None
            else initial_raw
        )
        payload = {
            "parameter_names": list(active_theta_names(self.config)),
            "initial_raw_thetas": initial_raw.tolist(),
            "final_raw_thetas": final_raw.tolist(),
            "initial_train_size": int(initial_raw.shape[0]),
            "final_train_size": int(final_raw.shape[0]),
            "selected_raw_thetas_by_iteration": [
                np.asarray(item.selected_raw_thetas, dtype=np.float64).tolist()
                for item in context.iteration_records
            ],
            "selected_unit_thetas_by_iteration": [
                np.asarray(item.selected_unit_thetas, dtype=np.float64).tolist()
                for item in context.iteration_records
            ],
        }
        path = run_process_path(context.run_dir, "training_point_summary.json", create=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_hifi_bank(self, context: ValidationContext) -> Path | None:
        dataset = context.dataset
        if dataset is None:
            return None
        path = run_process_path(context.run_dir, "hifi_bank.npz", create=True)
        payload = {
            "train_thetas": np.asarray(dataset.raw_thetas, dtype=np.float64),
            "train_k_bins": np.asarray(dataset.k_bins, dtype=np.float64),
            "train_nonlin_pk": np.asarray(dataset.pk_batch, dtype=np.float64),
        }
        if dataset.p_linear_batch is not None:
            payload["train_linear_pk"] = np.asarray(dataset.p_linear_batch, dtype=np.float64)
        np.savez_compressed(path, **payload)
        return path

    def _write_run_metadata(self, context: ValidationContext) -> None:
        self._flush_process_progress(context)
        results_path = run_results_path(context.run_dir, "run_metadata.json", create=True)
        results_path.write_text(
            json.dumps(context.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        latest_run_path = Path(self.config.project_root) / self.config.reports_dir / "latest_run.txt"
        latest_run_path.parent.mkdir(parents=True, exist_ok=True)
        latest_run_path.write_text(str(context.run_dir), encoding="utf-8")

    def _capture_dynamic_initial_state(self) -> dict[str, Any]:
        return {
            "m3_representation_band_weights": [
                float(value) for value in self.config.m3.representation_band_weights
            ],
            "band_pca_components": [
                int(value) for value in self.config.representation.band_pca_components
            ],
            "grid_fractions": [
                float(self.config.grids.low_k_fraction),
                float(self.config.grids.mid_k_fraction),
                float(self.config.grids.high_k_fraction),
            ],
        }

    def _restore_dynamic_initial_state(self, state: dict[str, Any]) -> None:
        weights = np.asarray(
            state.get("m3_representation_band_weights", ()),
            dtype=np.float64,
        ).reshape(-1)
        if weights.size and np.all(np.isfinite(weights)) and np.all(weights > 0.0):
            self.config.m3.representation_band_weights = tuple(float(value) for value in weights)
            self.config.m3.__post_init__()

        components = tuple(int(value) for value in state.get("band_pca_components", ()))
        if components and all(value >= 0 for value in components):
            self.config.representation.band_pca_components = components
            self.config.representation.__post_init__()

        grid_fractions = np.asarray(
            state.get("grid_fractions", ()),
            dtype=np.float64,
        ).reshape(-1)
        if (
            grid_fractions.shape == (3,)
            and np.all(np.isfinite(grid_fractions))
            and np.all(grid_fractions > 0.0)
        ):
            (
                self.config.grids.low_k_fraction,
                self.config.grids.mid_k_fraction,
                self.config.grids.high_k_fraction,
            ) = tuple(float(value) for value in grid_fractions)
            self.config.grids.__post_init__()

    def _complete_run_with_dynamic_reset(
        self,
        context: ValidationContext,
        dynamic_initial_state: dict[str, Any],
    ) -> ValidationContext:
        final_dynamic_state = self._capture_dynamic_initial_state()
        context.metadata["dynamic_initial_state_reset_after_completion"] = True
        context.metadata["dynamic_initial_state_restore_target"] = dict(dynamic_initial_state)
        context.metadata["dynamic_final_state_before_completion_reset"] = dict(final_dynamic_state)
        context.metadata["dynamic_reset_excluded_parameters"] = [
            "fastmock_bias.bias_weight",
            "lambda_bias_weight",
        ]
        self._restore_dynamic_initial_state(dynamic_initial_state)
        restored_state = self._capture_dynamic_initial_state()
        context.metadata["dynamic_state_after_completion_reset"] = dict(restored_state)
        context.metadata["m3_representation_band_weights"] = list(
            restored_state.get("m3_representation_band_weights", [])
        )
        context.metadata["band_pca_components"] = list(restored_state.get("band_pca_components", []))
        context.metadata["grid_fractions"] = list(restored_state.get("grid_fractions", []))
        self._write_run_metadata(context)
        return context

    @staticmethod
    def _dynamic_update_due(iteration_index: int, interval: int) -> bool:
        return int(interval) > 0 and int(iteration_index) > 1 and (int(iteration_index) - 1) % int(interval) == 0

    def _update_context_runtime_metadata(
        self,
        context: ValidationContext,
        *,
        dynamic_snapshot: dict[str, Any] | None = None,
    ) -> None:
        dataset = context.dataset
        emulator = context.emulator
        if dataset is None:
            return
        target_transform = resolve_target_transform_from_metadata(
            dataset.metadata,
            transform_family=str(self.config.representation.transform_family),
            anchor_mode=str(self.config.representation.anchor_mode),
        )
        context.metadata.update(
            {
                "target_transform": target_transform,
                "representation_transform_family": str(
                    dataset.metadata.get(
                        "representation_transform_family",
                        self.config.representation.transform_family,
                    )
                ),
                "representation_anchor_mode": str(
                    dataset.metadata.get(
                        "representation_anchor_mode",
                        self.config.representation.anchor_mode,
                    )
                ),
                "pca_scheme": str(dataset.metadata.get("pca_scheme", self.config.representation.pca_scheme)),
                "global_pca_components": int(self.config.representation.global_pca_components),
                "band_pca_components": [
                    int(value) for value in self.config.representation.band_pca_components
                ],
                "pca_layout": dict(dataset.metadata.get("pca_layout", {})),
                "m3_objective_mode": str(self.config.m3.objective_mode),
                "m3_representation_global_weight": float(self.config.m3.representation_global_weight),
                "m3_representation_band_weights": [
                    float(value) for value in self.config.m3.representation_band_weights
                ],
                "dynamic_preprocessing_enabled": bool(self.config.dynamic_preprocessing.enabled),
            }
        )
        if emulator is not None:
            context.metadata["lengthscale_upper_hit_summary"] = summarize_lengthscale_upper_hits(
                self.config,
                emulator,
            )
        if dynamic_snapshot is not None:
            context.metadata["dynamic_preprocessing_state"] = dict(dynamic_snapshot)

    def _resolve_dynamic_error_signal(
        self,
        emulator: EmulatorState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dynamic = self.config.dynamic_preprocessing
        preferred_source = str(dynamic.error_source)
        if preferred_source == "validation_relative_error":
            try:
                validation_bank = self.validation_bank or get_or_create_validation_bank(
                    self.config,
                    self.camb_data_provider,
                    progress_callback=self._emit_progress,
                    force_rebuild=self.force_rebuild_cache,
                )
                self.validation_bank = validation_bank
                subset_size = int(dynamic.validation_subset_size)
                if subset_size > 0 and subset_size < int(validation_bank.raw_thetas.shape[0]):
                    subset = slice(0, subset_size)
                else:
                    subset = slice(None)
                prediction = predict_spectra(
                    emulator,
                    np.asarray(validation_bank.raw_thetas[subset], dtype=np.float64),
                    input_space="raw",
                    k_target=np.asarray(validation_bank.k_bins, dtype=np.float64),
                    p_linear_batch=(
                        None
                        if validation_bank.p_linear_batch is None
                        else np.asarray(validation_bank.p_linear_batch[subset], dtype=np.float64)
                    ),
                )
                errors = compute_band_relative_errors(
                    np.asarray(validation_bank.k_bins, dtype=np.float64),
                    np.asarray(validation_bank.p_nonlin_batch[subset], dtype=np.float64),
                    np.asarray(prediction.pk_mean, dtype=np.float64),
                    eps=float(self.config.eps_r),
                )
                return errors.astype(np.float64), {
                    "source": "validation_relative_error",
                    "validation_cache_path": str(validation_bank.npz_path),
                    "validation_points_used": int(prediction.pk_mean.shape[0]),
                    "sampling_method": str(validation_bank.metadata.get("sampling_method", "unknown")),
                }
            except Exception as exc:
                if not bool(dynamic.proxy_fallback_enabled):
                    raise
                fallback_meta = {
                    "source": "validation_relative_error_fallback",
                    "fallback_reason": str(exc),
                }
            else:
                fallback_meta = {}
        else:
            fallback_meta = {"source": "pca_sensitivity_proxy"}

        continuous_state = build_continuous_posterior_state(emulator)
        sensitivity = np.asarray(
            continuous_state.metadata.get("pca_band_sensitivity", []),
            dtype=np.float64,
        )
        proxy_scores = compute_proxy_band_scores_from_sensitivity(sensitivity)
        fallback_meta.setdefault("source", "pca_sensitivity_proxy")
        fallback_meta["sensitivity_matrix"] = sensitivity.tolist()
        return proxy_scores.astype(np.float64), fallback_meta

    def _resolve_dynamic_posterior_variance_signal(
        self,
        emulator: EmulatorState,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        dynamic = self.config.dynamic_preprocessing
        if str(dynamic.band_weight_balance_mode) != "core_posterior_variance":
            return None, {"source": "disabled_balance_mode"}
        if float(dynamic.posterior_variance_eta) <= 0.0:
            return None, {"source": "disabled_zero_eta"}
        probe_size = int(dynamic.posterior_variance_probe_size)
        if probe_size <= 0:
            return None, {"source": "disabled_zero_probe_size"}

        continuous_state = build_continuous_posterior_state(emulator)
        theta_dim = int(continuous_state.train_unit_thetas.shape[1])
        seed = int(self.config.random_seed + int(dynamic.posterior_variance_seed_offset))
        probe_unit = generate_unit_sobol_samples(
            theta_dim,
            probe_size,
            seed,
            scramble=True,
        )
        variance_eval = evaluate_continuous_variance(
            continuous_state,
            probe_unit,
            input_space="unit",
        )
        sensitivity = np.asarray(
            continuous_state.metadata.get("pca_band_sensitivity", []),
            dtype=np.float64,
        )
        posterior_variance = compute_band_posterior_variance_scores(
            np.asarray(variance_eval.pc_var, dtype=np.float64),
            sensitivity,
        )
        mean_variance = float(np.mean(posterior_variance))
        normalized = (
            np.ones_like(posterior_variance, dtype=np.float64)
            if not np.isfinite(mean_variance) or mean_variance <= 0.0
            else posterior_variance / mean_variance
        )
        return posterior_variance.astype(np.float64), {
            "source": "posterior_variance_probe",
            "probe_size": int(probe_size),
            "probe_seed": int(seed),
            "theta_dim": int(theta_dim),
            "raw_band_posterior_variance": posterior_variance.astype(np.float64).tolist(),
            "mean_normalized_band_posterior_variance": normalized.astype(np.float64).tolist(),
            "pca_band_sensitivity": sensitivity.astype(np.float64).tolist(),
        }

    def _apply_dynamic_preprocessing(
        self,
        context: ValidationContext,
        dataset: Module1Dataset,
        emulator: EmulatorState,
        *,
        iteration_index: int,
    ) -> tuple[Module1Dataset, EmulatorState, dict[str, Any]]:
        dynamic = self.config.dynamic_preprocessing
        band_count = len(PCA_BAND_LABELS)
        previous_state = dict(context.metadata.get("dynamic_preprocessing_state", {}))
        effective_band_weights = np.asarray(
            previous_state.get(
                "effective_band_weights",
                self.config.m3.representation_band_weights,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if effective_band_weights.shape != (band_count,):
            effective_band_weights = np.asarray(self.config.m3.representation_band_weights, dtype=np.float64)
        effective_band_components = tuple(
            int(value)
            for value in previous_state.get(
                "effective_band_pca_components",
                self.config.representation.band_pca_components,
            )
        )
        effective_grid_fractions = np.asarray(
            previous_state.get(
                "effective_grid_fractions",
                self.config.m3.representation_band_weights,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if effective_grid_fractions.shape != (band_count,):
            effective_grid_fractions = np.ones((band_count,), dtype=np.float64)
        effective_grid_fractions = effective_grid_fractions / float(np.sum(effective_grid_fractions))

        update_band_weights = bool(dynamic.enabled and dynamic.update_band_weights) and self._dynamic_update_due(
            iteration_index,
            int(dynamic.band_weight_update_interval),
        )
        update_band_components = bool(dynamic.enabled and dynamic.update_pca_allocation) and self._dynamic_update_due(
            iteration_index,
            int(dynamic.band_component_update_interval),
        )
        update_grid = bool(dynamic.enabled and dynamic.update_k_grid) and self._dynamic_update_due(
            iteration_index,
            int(dynamic.grid_update_interval),
        )

        error_signal = np.ones((band_count,), dtype=np.float64)
        error_signal_metadata: dict[str, Any] = {
            "source": "static_default",
        }
        posterior_variance_signal: np.ndarray | None = None
        posterior_variance_metadata: dict[str, Any] = {
            "source": "not_requested",
        }
        weight_update_metadata: dict[str, Any] = {}
        rebuild_required = False
        if update_band_weights or update_band_components or update_grid:
            error_signal, error_signal_metadata = self._resolve_dynamic_error_signal(emulator)
        if update_band_weights:
            posterior_variance_signal, posterior_variance_metadata = (
                self._resolve_dynamic_posterior_variance_signal(emulator)
            )

        if update_band_weights:
            effective_band_weights, weight_update_metadata = update_band_weights_from_errors(
                error_signal,
                effective_band_weights,
                gamma=float(dynamic.weight_gamma),
                rho=float(dynamic.weight_rho),
                weight_min=float(dynamic.weight_min),
                weight_max=float(dynamic.weight_max),
                balance_mode=str(dynamic.band_weight_balance_mode),
                posterior_variance_by_band=posterior_variance_signal,
                posterior_variance_gamma=float(dynamic.posterior_variance_gamma),
                posterior_variance_eta=float(dynamic.posterior_variance_eta),
                error_signal_eta=float(dynamic.error_signal_eta),
                core_band_indices=tuple(int(value) for value in dynamic.core_band_indices),
                core_error_good=float(dynamic.core_error_good),
                core_error_bad=float(dynamic.core_error_bad),
                core_gate_floor=float(dynamic.core_gate_floor),
                core_gate_ceiling=float(dynamic.core_gate_ceiling),
                core_priority=tuple(float(value) for value in dynamic.core_priority),
                release_priority=tuple(float(value) for value in dynamic.release_priority),
                band_weight_prior=tuple(float(value) for value in dynamic.band_weight_prior),
                prior_eta=float(dynamic.weight_prior_eta),
                return_metadata=True,
            )
            if not np.allclose(
                effective_band_weights,
                np.asarray(self.config.m3.representation_band_weights, dtype=np.float64),
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                self.config.m3.representation_band_weights = tuple(
                    float(value) for value in effective_band_weights
                )
                self.config.m3.__post_init__()

        if update_band_components and self.config.representation.pca_scheme in {
            "bandwise_pca",
            "global_plus_band_residual_pca",
        }:
            residual_budget = int(self.config.gp.pca_components)
            if self.config.representation.pca_scheme == "global_plus_band_residual_pca":
                residual_budget = max(
                    0,
                    int(self.config.gp.pca_components)
                    - int(self.config.representation.global_pca_components),
                )
            new_band_components = allocate_band_components(
                residual_budget,
                effective_band_weights,
                effective_band_components,
                allocation_lambda=float(dynamic.allocation_lambda),
                min_band_components=int(dynamic.min_band_components),
                max_delta_per_update=int(dynamic.max_component_delta_per_update),
            )
            if tuple(new_band_components) != tuple(int(v) for v in self.config.representation.band_pca_components):
                self.config.representation.band_pca_components = tuple(int(value) for value in new_band_components)
                self.config.representation.__post_init__()
                effective_band_components = tuple(int(value) for value in new_band_components)
                rebuild_required = True

        if update_grid:
            new_grid_fractions = merge_band_weights_to_grid_fractions(
                effective_band_weights,
                effective_grid_fractions,
                rho=float(dynamic.weight_rho),
            )
            current_grid = (
                float(self.config.grids.low_k_fraction),
                float(self.config.grids.mid_k_fraction),
                float(self.config.grids.high_k_fraction),
            )
            if not np.allclose(
                np.asarray(new_grid_fractions, dtype=np.float64),
                np.asarray(effective_grid_fractions, dtype=np.float64),
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                effective_grid_fractions = np.asarray(new_grid_fractions, dtype=np.float64)
                rebuild_required = True

        if rebuild_required:
            dataset = build_dataset_from_spectrum_bank(
                self.config,
                np.asarray(dataset.raw_thetas, dtype=np.float64),
                np.asarray(dataset.k_bins, dtype=np.float64),
                np.asarray(dataset.pk_batch, dtype=np.float64),
                p_linear_batch=(
                    None if dataset.p_linear_batch is None else np.asarray(dataset.p_linear_batch, dtype=np.float64)
                ),
            )
            emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)

        snapshot = {
            "iteration_index": int(iteration_index),
            "enabled": bool(dynamic.enabled),
            "update_band_weights": bool(update_band_weights),
            "update_band_components": bool(update_band_components),
            "update_grid": bool(update_grid),
            "error_signal_source": str(error_signal_metadata.get("source", "static_default")),
            "error_signal_by_band": np.asarray(error_signal, dtype=np.float64).tolist(),
            "error_signal_metadata": dict(error_signal_metadata),
            "posterior_variance_by_band": (
                None
                if posterior_variance_signal is None
                else np.asarray(posterior_variance_signal, dtype=np.float64).tolist()
            ),
            "posterior_variance_metadata": dict(posterior_variance_metadata),
            "band_weight_update_metadata": dict(weight_update_metadata),
            "band_weight_prior": [float(value) for value in dynamic.band_weight_prior],
            "weight_prior_eta": float(dynamic.weight_prior_eta),
            "effective_band_weights": np.asarray(effective_band_weights, dtype=np.float64).tolist(),
            "effective_band_pca_components": [int(value) for value in self.config.representation.band_pca_components],
            "effective_grid_fractions": [
                float(value) for value in effective_grid_fractions
            ],
            "effective_global_pca_components": int(self.config.representation.global_pca_components),
            "rebuild_required": bool(rebuild_required),
            "pca_scheme": str(self.config.representation.pca_scheme),
        }
        history = list(context.metadata.get("dynamic_preprocessing_history", []))
        history.append(dict(snapshot))
        context.metadata["dynamic_preprocessing_history"] = history
        context.metadata["dynamic_preprocessing_state"] = dict(snapshot)
        return dataset, emulator, snapshot

    @staticmethod
    def _integrated_focus_relative_error_per_sample(
        k_bins: np.ndarray,
        p_true_batch: np.ndarray,
        p_pred_batch: np.ndarray,
        *,
        k_min: float,
        k_max: float,
        eps: float,
    ) -> np.ndarray:
        k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        true_arr = np.asarray(p_true_batch, dtype=np.float64)
        pred_arr = np.asarray(p_pred_batch, dtype=np.float64)
        if true_arr.shape != pred_arr.shape:
            raise ValueError(
                "p_true_batch and p_pred_batch must align for validation probe "
                f"errors, got {true_arr.shape} vs {pred_arr.shape}."
            )
        if true_arr.ndim != 2 or true_arr.shape[1] != k_arr.shape[0]:
            raise ValueError(
                "validation probe spectra must be 2D and align with k_bins, "
                f"got {true_arr.shape} vs {k_arr.shape}."
            )
        mask = (k_arr >= float(k_min)) & (k_arr <= float(k_max))
        if not np.any(mask):
            mask = np.ones_like(k_arr, dtype=bool)
        logk = np.log10(np.maximum(k_arr, 1.0e-30))
        weights = np.empty_like(logk, dtype=np.float64)
        weights[0] = 0.5 * (logk[1] - logk[0]) if logk.size > 1 else 1.0
        weights[-1] = 0.5 * (logk[-1] - logk[-2]) if logk.size > 1 else 1.0
        if logk.size > 2:
            weights[1:-1] = 0.5 * (logk[2:] - logk[:-2])
        band_weights = np.maximum(weights[mask], 0.0)
        band_weights = band_weights / max(float(np.sum(band_weights)), 1.0e-30)
        relative_error = np.abs(pred_arr - true_arr) / np.maximum(
            np.abs(true_arr),
            float(max(eps, 1.0e-30)),
        )
        return np.asarray(relative_error[:, mask] @ band_weights, dtype=np.float64)

    def _resolve_p68_validation_probe_shell(
        self,
        emulator: EmulatorState,
    ) -> dict[str, Any]:
        m3 = self.config.m3
        if str(m3.acquisition_p68_set_rerank_risk_mode) != "validation_probe_shell":
            return {"enabled": False, "reason": "risk_mode_disabled"}
        if int(m3.acquisition_p68_validation_probe_size) <= 0:
            return {"enabled": False, "reason": "probe_size_zero"}
        if int(m3.acquisition_p68_set_rerank_top_k) <= 0 or float(m3.acquisition_p68_set_rerank_p68_weight) <= 0.0:
            return {"enabled": False, "reason": "p68_rerank_disabled"}

        validation_bank = self.validation_bank or get_or_create_validation_bank(
            self.config,
            self.camb_data_provider,
            progress_callback=self._emit_progress,
            force_rebuild=self.force_rebuild_cache,
        )
        self.validation_bank = validation_bank
        probe_size = min(
            int(m3.acquisition_p68_validation_probe_size),
            int(validation_bank.raw_thetas.shape[0]),
        )
        if probe_size <= 1:
            return {"enabled": False, "reason": "not_enough_validation_points"}

        probe_slice = slice(0, probe_size)
        probe_raw = np.asarray(validation_bank.raw_thetas[probe_slice], dtype=np.float64)
        prediction = predict_spectra(
            emulator,
            probe_raw,
            input_space="raw",
            k_target=np.asarray(validation_bank.k_bins, dtype=np.float64),
            p_linear_batch=(
                None
                if validation_bank.p_linear_batch is None
                else np.asarray(validation_bank.p_linear_batch[probe_slice], dtype=np.float64)
            ),
        )
        errors = self._integrated_focus_relative_error_per_sample(
            np.asarray(validation_bank.k_bins, dtype=np.float64),
            np.asarray(validation_bank.p_nonlin_batch[probe_slice], dtype=np.float64),
            np.asarray(prediction.pk_mean, dtype=np.float64),
            k_min=float(m3.acquisition_p68_validation_probe_focus_k_min),
            k_max=float(m3.acquisition_p68_validation_probe_focus_k_max),
            eps=float(self.config.eps_r),
        )
        q68 = float(np.percentile(errors, 68))
        shell_width = float(m3.acquisition_p68_validation_probe_shell_width)
        scale = max(shell_width * max(q68, float(self.config.eps_r)), 1.0e-12)
        shell = np.exp(-0.5 * np.square((errors - q68) / scale))
        floor = float(m3.acquisition_p68_validation_probe_min_weight)
        weights = floor + (1.0 - floor) * shell
        weights = weights / max(float(np.mean(weights)), 1.0e-30)
        return {
            "enabled": True,
            "mode": "validation_probe_shell",
            "probe_size": int(probe_size),
            "source": "validation_bank_first_n",
            "validation_cache_path": str(validation_bank.npz_path),
            "sampling_method": str(validation_bank.metadata.get("sampling_method", "unknown")),
            "focus_k_min": float(m3.acquisition_p68_validation_probe_focus_k_min),
            "focus_k_max": float(m3.acquisition_p68_validation_probe_focus_k_max),
            "shell_width_fraction": shell_width,
            "min_weight": floor,
            "error_q68": q68,
            "error_min": float(np.min(errors)),
            "error_max": float(np.max(errors)),
            "error_mean": float(np.mean(errors)),
            "probe_raw_thetas": probe_raw.astype(np.float64).tolist(),
            "probe_unit_thetas": normalize_theta_batch(
                probe_raw,
                emulator.theta_bounds,
            ).astype(np.float64).tolist(),
            "probe_errors": errors.astype(np.float64).tolist(),
            "probe_weights": weights.astype(np.float64).tolist(),
        }

    def _build_module3_input(
        self,
        emulator: EmulatorState,
        *,
        iteration_index: int,
        dynamic_snapshot: dict[str, Any] | None = None,
    ) -> Module3ContinuousInput:
        continuous_state = build_continuous_posterior_state(emulator)
        lengthscale_hit_summary = summarize_lengthscale_upper_hits(self.config, emulator)
        pca_band_diagnostics = build_iteration_pca_band_diagnostics(
            self.config,
            continuous_state,
            iteration_index=int(iteration_index),
        )
        effective_band_weights = (
            list(dynamic_snapshot.get("effective_band_weights", []))
            if dynamic_snapshot is not None
            else [float(value) for value in self.config.m3.representation_band_weights]
        )
        if len(effective_band_weights) != len(PCA_BAND_LABELS):
            effective_band_weights = [float(value) for value in self.config.m3.representation_band_weights]
        validation_probe_shell = self._resolve_p68_validation_probe_shell(emulator)
        transform_family = str(
            emulator.dataset.metadata.get(
                "representation_transform_family",
                self.config.representation.transform_family,
            )
        ).strip().lower()
        if transform_family == "logdiff":
            target_transform = resolve_target_transform_from_metadata(
                emulator.dataset.metadata,
                transform_family=str(self.config.representation.transform_family),
                anchor_mode=str(self.config.representation.anchor_mode),
            )
            projection = build_logdiff_projected_component_weights(
                np.asarray(emulator.dataset.pca_components, dtype=np.float64),
                np.asarray(emulator.dataset.k_bins, dtype=np.float64),
                band_multipliers=[1.0 for _ in PCA_BAND_LABELS],
                base_band_levels=effective_band_weights,
                component_groups=continuous_state.metadata.get("representation_component_groups", []),
                global_weight=float(self.config.m3.representation_global_weight),
            )
            logdiff_projection_details = {
                **dict(projection["details"]),
                "representation_global_weight": float(self.config.m3.representation_global_weight),
                "representation_band_weights": [float(value) for value in effective_band_weights],
                "pca_scheme": str(emulator.dataset.metadata.get("pca_scheme", "global_pca")),
                "target_transform": target_transform,
            }
            logdiff_projected_component_weights = np.asarray(
                projection["component_weights"],
                dtype=np.float64,
            )
        else:
            logdiff_projection_details = {}
            logdiff_projected_component_weights = np.empty((0,), dtype=np.float64)
        continuous_state.metadata.update(
            {
                "iteration_index": int(iteration_index),
                "pca_band_diagnostics": dict(pca_band_diagnostics),
                "m3_effective_global_weight": float(self.config.m3.representation_global_weight),
                "m3_effective_band_weights": [float(value) for value in effective_band_weights],
                "dynamic_preprocessing_snapshot": (
                    {} if dynamic_snapshot is None else dict(dynamic_snapshot)
                ),
                "gp_lengthscale_upper_hit_summary": dict(lengthscale_hit_summary),
                "logdiff_projected_component_weights": (
                    logdiff_projected_component_weights.astype(np.float64).tolist()
                ),
                "logdiff_projected_component_weight_details": dict(logdiff_projection_details),
                "p68_validation_probe_shell": dict(validation_probe_shell),
            }
        )
        self._emit_progress("module2_pca_band_sensitivity", 1, 1)
        return Module3ContinuousInput(
            continuous_state=continuous_state,
            iteration_index=int(iteration_index),
            metadata={
                "iteration_index": int(iteration_index),
                "batch_size": int(self.config.sampling.batch_size),
                "train_size": int(continuous_state.train_unit_thetas.shape[0]),
                "pc_dim": int(len(continuous_state.gp_models)),
                "pca_band_diagnostics_available": True,
                "pca_band_weight_function": str(self.config.m3.weight_function),
                "pca_band_weight_temperature": float(self.config.m3.weight_temperature),
                "target_transform": resolve_target_transform_from_metadata(
                    emulator.dataset.metadata,
                    transform_family=str(self.config.representation.transform_family),
                    anchor_mode=str(self.config.representation.anchor_mode),
                ),
                "pca_scheme": str(emulator.dataset.metadata.get("pca_scheme", "global_pca")),
                "global_pca_components": int(self.config.representation.global_pca_components),
                "band_pca_components": [
                    int(value) for value in self.config.representation.band_pca_components
                ],
                "m3_effective_global_weight": float(self.config.m3.representation_global_weight),
                "m3_effective_band_weights": [float(value) for value in effective_band_weights],
                "gp_lengthscale_upper_hit_summary": dict(lengthscale_hit_summary),
                "dynamic_preprocessing_snapshot": (
                    {} if dynamic_snapshot is None else dict(dynamic_snapshot)
                ),
                "logdiff_projected_component_weights": (
                    logdiff_projected_component_weights.astype(np.float64).tolist()
                ),
                "logdiff_projected_component_weight_details": dict(logdiff_projection_details),
                "p68_validation_probe_shell": dict(validation_probe_shell),
                "pca_band_focus_top_components": list(
                    pca_band_diagnostics.get("focus_top_components", [])
                ),
            },
        )

    def _await_module3(
        self,
        context: ValidationContext,
        module3_input: Module3ContinuousInput,
        *,
        input_path: Path | None = None,
        summary_path: Path | None = None,
    ) -> ValidationContext:
        if input_path is None or summary_path is None:
            input_path, summary_path = self._serialize_module3_input(
                module3_input,
                context=context,
            )
        context.metadata.update(
            {
                "status": "awaiting_module3_selector",
                "pending_iteration": int(module3_input.iteration_index),
                "module3_continuous_input_artifact": str(input_path),
                "module3_continuous_summary_artifact": str(summary_path),
                "completed_iterations": int(len(context.iteration_records)),
            }
        )
        self._write_run_metadata(context)
        return context

    def _persist_module3_input_artifacts(
        self,
        context: ValidationContext,
        module3_input: Module3ContinuousInput,
    ) -> tuple[Path, Path]:
        input_path, summary_path = self._serialize_module3_input(
            module3_input,
            context=context,
        )
        records = list(context.metadata.get("module3_input_artifacts", []))
        records.append(
            {
                "iteration_index": int(module3_input.iteration_index),
                "state_path": str(input_path),
                "summary_path": str(summary_path),
            }
        )
        context.metadata["module3_input_artifacts"] = records
        context.metadata["latest_module3_input_artifact"] = str(input_path)
        context.metadata["latest_module3_summary_artifact"] = str(summary_path)
        return input_path, summary_path

    def _resolve_initial_dataset(self) -> Module1Dataset:
        if self.initial_dataset is not None:
            dataset = self.initial_dataset
            dataset.metadata.setdefault("source", "injected_dataset")
            return dataset
        if self.use_initial_training_cache:
            initial_bank = get_or_create_initial_training_bank(
                self.config,
                self.camb_data_provider,
                progress_callback=self._emit_progress,
                force_rebuild=self.force_rebuild_cache,
            )
            dataset = build_dataset_from_spectrum_bank(
                self.config,
                initial_bank.raw_thetas,
                initial_bank.k_bins,
                initial_bank.p_nonlin_batch,
                p_linear_batch=initial_bank.p_linear_batch,
            )
            dataset.metadata.update(
                {
                    "source": "initial_training_cache",
                    "cache_name": str(initial_bank.name),
                    "cache_path": str(initial_bank.npz_path),
                    "cache_status": str(initial_bank.metadata.get("cache_status", "unknown")),
                }
            )
            return dataset
        dataset = build_initial_dataset(
            self.config,
            self.camb_data_provider,
            asset_version="active_learning_initial",
            progress_callback=self._emit_progress,
        )
        dataset.metadata.setdefault("source", "module1_rebuild")
        return dataset

    def resume(self, run_dir: str | Path) -> ValidationContext:
        resolved_run_dir = Path(run_dir).resolve()
        results_dir, process_dir = ensure_run_artifact_layout(resolved_run_dir)
        dynamic_initial_state = self._capture_dynamic_initial_state()
        checkpoint_artifacts = self._discover_module3_input_artifacts(resolved_run_dir)
        if not checkpoint_artifacts:
            raise FileNotFoundError(
                f"No module3 checkpoint artifacts were found under {process_dir}."
            )

        total_iterations = int(self.config.sampling.iterations)
        initial_artifact = checkpoint_artifacts[0]
        initial_summary_path = initial_artifact.get("summary_path")
        if initial_summary_path is None:
            raise FileNotFoundError("Initial checkpoint summary is missing; cannot resume.")
        initial_summary = self._load_module3_summary(Path(initial_summary_path))
        initial_raw_thetas = np.asarray(initial_summary["train_raw_thetas"], dtype=np.float64)

        recovered_records = self._load_saved_iteration_records(resolved_run_dir)
        if not recovered_records:
            _, recovered_records = self._reconstruct_completed_iteration_records(
                checkpoint_artifacts=checkpoint_artifacts
            )
        recovered_records = sorted(
            recovered_records,
            key=lambda item: int(item.iteration_index),
        )

        highest_completed_iteration = max(
            (int(item.iteration_index) for item in recovered_records),
            default=0,
        )
        pending_artifact = checkpoint_artifacts[-1]
        highest_checkpoint_iteration = int(pending_artifact["iteration_index"])
        if highest_completed_iteration > highest_checkpoint_iteration:
            raise ValueError(
                "Recovered iteration history is ahead of the latest module3 checkpoint: "
                f"{highest_completed_iteration} > {highest_checkpoint_iteration}."
            )
        if highest_checkpoint_iteration > total_iterations:
            raise ValueError(
                "Checkpoint iteration exceeds configured iteration count: "
                f"{highest_checkpoint_iteration} > {total_iterations}."
            )

        pending_iteration: int | None = None
        pending_module3_input: Module3ContinuousInput | None = None
        pending_state_path: Path | None = None
        pending_summary_path: Path | None = None

        if highest_checkpoint_iteration > highest_completed_iteration:
            pending_iteration = int(highest_checkpoint_iteration)
            pending_state = pending_artifact.get("state_path")
            pending_summary = pending_artifact.get("summary_path")
            if pending_state is None or pending_summary is None:
                raise FileNotFoundError(
                    f"Pending iteration {pending_iteration} is missing its checkpoint artifact pair."
                )
            pending_state_path = Path(pending_state)
            pending_summary_path = Path(pending_summary)
            pending_module3_input = self._load_module3_input(pending_state_path)
            pending_raw_thetas = np.asarray(
                pending_module3_input.continuous_state.train_raw_thetas,
                dtype=np.float64,
            )
            dataset = self._rebuild_dataset_from_checkpoint(
                run_dir=resolved_run_dir,
                raw_thetas=pending_raw_thetas,
                iteration_index=pending_iteration,
            )
        else:
            next_iteration = int(highest_completed_iteration + 1)
            completed_raw_thetas = self._reconstruct_training_raw_from_records(
                initial_raw_thetas,
                recovered_records,
            )
            dataset = self._rebuild_dataset_from_checkpoint(
                run_dir=resolved_run_dir,
                raw_thetas=completed_raw_thetas,
                iteration_index=max(1, highest_completed_iteration),
            )
            if next_iteration <= total_iterations:
                pending_iteration = int(next_iteration)

        context = ValidationContext(
            config=self.config,
            run_dir=resolved_run_dir,
            results_dir=results_dir,
            process_dir=process_dir,
            metadata={
                "mode": "active_learning_emulator",
                "status": "resuming",
                "project_root": self.config.project_root,
                "spectrum_type": resolve_data_source(self.config).spectrum_type,
                "data_source": resolve_data_source(self.config).name,
                "parameter_space": resolve_data_source(self.config).parameter_space,
                "parameter_names": list(active_theta_names(self.config)),
                "initial_sobol_points": int(self.config.sampling.initial_sobol_points),
                "iterations": total_iterations,
                "batch_size": int(self.config.sampling.batch_size),
                "dynamic_initial_state": dict(dynamic_initial_state),
                "module3_input_mode": "continuous_only",
                "module3_selector": (
                    type(self.batch_selector).__name__
                    if self.batch_selector is not None
                    else InterfaceOnlySelector.__name__
                ),
                "initial_dataset_source": "resume_training_cache",
                "resume_run_dir": str(resolved_run_dir),
                "resume_pending_iteration": pending_iteration,
                "resume_recovered_iterations": int(len(recovered_records)),
                "module3_input_artifacts": checkpoint_artifacts,
                "latest_module3_input_artifact": (
                    str(pending_state_path) if pending_state_path is not None else None
                ),
                "latest_module3_summary_artifact": (
                    str(pending_summary_path) if pending_summary_path is not None else None
                ),
            },
        )
        context.initial_raw_thetas = np.asarray(initial_raw_thetas, dtype=np.float64)
        context.dataset = dataset
        context.iteration_records = list(recovered_records)

        emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)
        context.emulator = emulator
        self._update_context_runtime_metadata(context)

        if pending_iteration is not None and pending_module3_input is None:
            dataset, emulator, dynamic_snapshot = self._apply_dynamic_preprocessing(
                context,
                dataset,
                emulator,
                iteration_index=int(pending_iteration),
            )
            context.dataset = dataset
            context.emulator = emulator
            self._update_context_runtime_metadata(context, dynamic_snapshot=dynamic_snapshot)
            pending_module3_input = self._build_module3_input(
                emulator,
                iteration_index=int(pending_iteration),
                dynamic_snapshot=dynamic_snapshot,
            )
            pending_state_path, pending_summary_path = self._persist_module3_input_artifacts(
                context,
                pending_module3_input,
            )
            context.metadata["latest_module3_input_artifact"] = str(pending_state_path)
            context.metadata["latest_module3_summary_artifact"] = str(pending_summary_path)

        if pending_iteration is None:
            validation_dir = run_results_subdir(resolved_run_dir, "active_learning_validation", create=True)
            validation = evaluate_emulator_on_validation_set(
                self.config,
                self.camb_data_provider,
                emulator,
                validation_dir,
                asset_version="active_learning_validation",
                progress_callback=self._emit_progress,
                metadata={
                    "mode": "active_learning_validation",
                    "train_size": int(dataset.raw_thetas.shape[0]),
                    "resumed_from_checkpoint": True,
                },
                validation_bank=self.validation_bank,
                force_rebuild_validation_cache=self.force_rebuild_cache,
            )
            context.validation = validation

            if self.config.gp_baseline.enabled:
                run_standard_sobol_gp_baseline(
                    run_dir=resolved_run_dir,
                    config=self.config,
                    test_set_results_path=validation.test_set_results_path,
                    spectrum_type=str(resolve_data_source(self.config).spectrum_type),
                    camb_data_provider=self.camb_data_provider,
                    output_subdir=self.config.gp_baseline.output_subdir,
                    n_train_points=int(self.config.gp_baseline.train_points),
                    progress_callback=self._emit_progress,
                    force_rebuild_training_cache=self.force_rebuild_cache,
                )

            for name, hook in self.extension_hooks.items():
                output = hook(context)
                context.extension_outputs[name] = dict(output or {})

            context.metadata.update(
                {
                    "status": "completed",
                    "completed_iterations": int(len(context.iteration_records)),
                    "final_train_size": int(dataset.raw_thetas.shape[0]),
                    "validation_results": str(validation.test_set_results_path),
                    "registered_extensions": sorted(self.extension_hooks.keys()),
                    "resumed_from_checkpoint": True,
                }
            )
            return self._complete_run_with_dynamic_reset(context, dynamic_initial_state)

        if self.batch_selector is None and pending_module3_input is not None:
            return self._await_module3(
                context,
                pending_module3_input,
                input_path=pending_state_path,
                summary_path=pending_summary_path,
            )

        selection = select_next_batch(
            self.config,
            pending_module3_input,
            selector=self.batch_selector,
            progress_callback=self._emit_progress,
        )
        dataset = extend_dataset(
            self.config,
            self.camb_data_provider,
            dataset,
            selection.selected_raw_thetas,
            asset_version=f"active_learning_iter_{int(pending_iteration):02d}",
            progress_callback=self._emit_progress,
        )
        emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)
        pending_record = IterationRecord(
            iteration_index=int(pending_iteration),
            train_size_before=int(dataset.raw_thetas.shape[0] - selection.selected_raw_thetas.shape[0]),
            train_size_after=int(dataset.raw_thetas.shape[0]),
            selected_raw_thetas=np.asarray(selection.selected_raw_thetas, dtype=np.float64),
            selected_unit_thetas=np.asarray(selection.selected_unit_thetas, dtype=np.float64),
            selected_source_pc=np.asarray(selection.selected_source_pc, dtype=np.int64),
            selected_scores=np.asarray(selection.selected_scores, dtype=np.float64),
            metadata={
                **dict(selection.metadata),
                "pca_band_diagnostics": dict(
                    pending_module3_input.continuous_state.metadata.get("pca_band_diagnostics", {})
                ),
                "dynamic_preprocessing_snapshot": dict(
                    pending_module3_input.continuous_state.metadata.get(
                        "dynamic_preprocessing_snapshot",
                        {},
                    )
                ),
                "gp_lengthscale_upper_hit_summary": dict(
                    pending_module3_input.continuous_state.metadata.get(
                        "gp_lengthscale_upper_hit_summary",
                        {},
                    )
                ),
                "module3_input_artifact": str(pending_state_path),
                "module3_summary_artifact": str(pending_summary_path),
                "resume_pending_iteration": int(pending_iteration),
            },
        )
        context.iteration_records.append(pending_record)
        context.dataset = dataset
        context.emulator = emulator
        context.metadata.update(
            {
                "status": "resuming",
                "completed_iterations": int(len(context.iteration_records)),
                "resume_pending_iteration": int(pending_iteration),
            }
        )
        self._save_iteration_progress(context, pending_record)

        for iteration_index in range(int(pending_iteration) + 1, total_iterations + 1):
            dataset, emulator, dynamic_snapshot = self._apply_dynamic_preprocessing(
                context,
                dataset,
                emulator,
                iteration_index=iteration_index,
            )
            context.dataset = dataset
            context.emulator = emulator
            self._update_context_runtime_metadata(context, dynamic_snapshot=dynamic_snapshot)
            module3_input = self._build_module3_input(
                emulator,
                iteration_index=iteration_index,
                dynamic_snapshot=dynamic_snapshot,
            )
            input_path, summary_path = self._persist_module3_input_artifacts(
                context,
                module3_input,
            )
            selection = select_next_batch(
                self.config,
                module3_input,
                selector=self.batch_selector,
                progress_callback=self._emit_progress,
            )
            dataset = extend_dataset(
                self.config,
                self.camb_data_provider,
                dataset,
                selection.selected_raw_thetas,
                asset_version=f"active_learning_iter_{iteration_index:02d}",
                progress_callback=self._emit_progress,
            )
            emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)
            record = IterationRecord(
                iteration_index=iteration_index,
                train_size_before=int(dataset.raw_thetas.shape[0] - selection.selected_raw_thetas.shape[0]),
                train_size_after=int(dataset.raw_thetas.shape[0]),
                selected_raw_thetas=np.asarray(selection.selected_raw_thetas, dtype=np.float64),
                selected_unit_thetas=np.asarray(selection.selected_unit_thetas, dtype=np.float64),
                selected_source_pc=np.asarray(selection.selected_source_pc, dtype=np.int64),
                selected_scores=np.asarray(selection.selected_scores, dtype=np.float64),
                metadata={
                    **dict(selection.metadata),
                    "pca_band_diagnostics": dict(
                        module3_input.continuous_state.metadata.get("pca_band_diagnostics", {})
                    ),
                    "dynamic_preprocessing_snapshot": dict(
                        module3_input.continuous_state.metadata.get(
                            "dynamic_preprocessing_snapshot",
                            {},
                        )
                    ),
                    "gp_lengthscale_upper_hit_summary": dict(
                        module3_input.continuous_state.metadata.get(
                            "gp_lengthscale_upper_hit_summary",
                            {},
                        )
                    ),
                    "module3_input_artifact": str(input_path),
                    "module3_summary_artifact": str(summary_path),
                },
            )
            context.iteration_records.append(record)
            context.dataset = dataset
            context.emulator = emulator
            context.metadata.update(
                {
                    "status": "resuming",
                    "completed_iterations": int(len(context.iteration_records)),
                    "resume_pending_iteration": iteration_index,
                }
            )
            self._save_iteration_progress(context, record)

        validation_dir = run_results_subdir(resolved_run_dir, "active_learning_validation", create=True)
        validation = evaluate_emulator_on_validation_set(
            self.config,
            self.camb_data_provider,
            emulator,
            validation_dir,
            asset_version="active_learning_validation",
            progress_callback=self._emit_progress,
            metadata={
                "mode": "active_learning_validation",
                "train_size": int(dataset.raw_thetas.shape[0]),
                "resumed_from_checkpoint": True,
            },
            validation_bank=self.validation_bank,
            force_rebuild_validation_cache=self.force_rebuild_cache,
        )
        context.validation = validation

        if self.config.gp_baseline.enabled:
            run_standard_sobol_gp_baseline(
                run_dir=resolved_run_dir,
                config=self.config,
                test_set_results_path=validation.test_set_results_path,
                spectrum_type=str(resolve_data_source(self.config).spectrum_type),
                camb_data_provider=self.camb_data_provider,
                output_subdir=self.config.gp_baseline.output_subdir,
                n_train_points=int(self.config.gp_baseline.train_points),
                progress_callback=self._emit_progress,
                force_rebuild_training_cache=self.force_rebuild_cache,
            )

        for name, hook in self.extension_hooks.items():
            output = hook(context)
            context.extension_outputs[name] = dict(output or {})

        context.metadata.update(
            {
                "status": "completed",
                "completed_iterations": int(len(context.iteration_records)),
                "final_train_size": int(dataset.raw_thetas.shape[0]),
                "validation_results": str(validation.test_set_results_path),
                "registered_extensions": sorted(self.extension_hooks.keys()),
                "resumed_from_checkpoint": True,
            }
        )
        return self._complete_run_with_dynamic_reset(context, dynamic_initial_state)

    def run(self) -> ValidationContext:
        run_dir = self.create_run_dir()
        results_dir, process_dir = ensure_run_artifact_layout(run_dir)
        dynamic_initial_state = self._capture_dynamic_initial_state()
        context = ValidationContext(
            config=self.config,
            run_dir=run_dir,
            results_dir=results_dir,
            process_dir=process_dir,
            metadata={
                "mode": "active_learning_emulator",
                "status": "initializing",
                "project_root": self.config.project_root,
                "spectrum_type": resolve_data_source(self.config).spectrum_type,
                "data_source": resolve_data_source(self.config).name,
                "parameter_space": resolve_data_source(self.config).parameter_space,
                "parameter_names": list(active_theta_names(self.config)),
                "initial_sobol_points": int(self.config.sampling.initial_sobol_points),
                "iterations": int(self.config.sampling.iterations),
                "batch_size": int(self.config.sampling.batch_size),
                "dynamic_initial_state": dict(dynamic_initial_state),
                "module3_input_mode": "continuous_only",
                "module3_selector": (
                    type(self.batch_selector).__name__
                    if self.batch_selector is not None
                    else InterfaceOnlySelector.__name__
                ),
                "initial_dataset_source": "pending",
            },
        )

        dataset = self._resolve_initial_dataset()
        emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)
        initial_raw_thetas = np.asarray(dataset.raw_thetas, dtype=np.float64).copy()
        context.initial_raw_thetas = initial_raw_thetas
        context.dataset = dataset
        context.emulator = emulator
        context.metadata["initial_dataset_source"] = str(
            dataset.metadata.get("source", "module1_rebuild")
        )
        self._update_context_runtime_metadata(context)
        if "cache_path" in dataset.metadata:
            context.metadata["initial_cache_path"] = str(dataset.metadata["cache_path"])
        if "cache_status" in dataset.metadata:
            context.metadata["initial_cache_status"] = str(dataset.metadata["cache_status"])
        if "cache_name" in dataset.metadata:
            context.metadata["initial_cache_name"] = str(dataset.metadata["cache_name"])

        for iteration_index in range(1, int(self.config.sampling.iterations) + 1):
            dataset, emulator, dynamic_snapshot = self._apply_dynamic_preprocessing(
                context,
                dataset,
                emulator,
                iteration_index=iteration_index,
            )
            context.dataset = dataset
            context.emulator = emulator
            self._update_context_runtime_metadata(context, dynamic_snapshot=dynamic_snapshot)
            module3_input = self._build_module3_input(
                emulator,
                iteration_index=iteration_index,
                dynamic_snapshot=dynamic_snapshot,
            )
            input_path, summary_path = self._persist_module3_input_artifacts(
                context,
                module3_input,
            )
            if self.batch_selector is None:
                return self._await_module3(
                    context,
                    module3_input,
                    input_path=input_path,
                    summary_path=summary_path,
                )

            selection = select_next_batch(
                self.config,
                module3_input,
                selector=self.batch_selector,
                progress_callback=self._emit_progress,
            )
            dataset = extend_dataset(
                self.config,
                self.camb_data_provider,
                dataset,
                selection.selected_raw_thetas,
                asset_version=f"active_learning_iter_{iteration_index:02d}",
                progress_callback=self._emit_progress,
            )
            emulator = fit_emulator(self.config, dataset, progress_callback=self._emit_progress)
            record = IterationRecord(
                iteration_index=iteration_index,
                train_size_before=int(dataset.raw_thetas.shape[0] - selection.selected_raw_thetas.shape[0]),
                train_size_after=int(dataset.raw_thetas.shape[0]),
                selected_raw_thetas=np.asarray(selection.selected_raw_thetas, dtype=np.float64),
                selected_unit_thetas=np.asarray(selection.selected_unit_thetas, dtype=np.float64),
                selected_source_pc=np.asarray(selection.selected_source_pc, dtype=np.int64),
                selected_scores=np.asarray(selection.selected_scores, dtype=np.float64),
                metadata={
                    **dict(selection.metadata),
                    "pca_band_diagnostics": dict(
                        module3_input.continuous_state.metadata.get("pca_band_diagnostics", {})
                    ),
                    "dynamic_preprocessing_snapshot": dict(
                        module3_input.continuous_state.metadata.get(
                            "dynamic_preprocessing_snapshot",
                            {},
                        )
                    ),
                    "gp_lengthscale_upper_hit_summary": dict(
                        module3_input.continuous_state.metadata.get(
                            "gp_lengthscale_upper_hit_summary",
                            {},
                        )
                    ),
                    "module3_input_artifact": str(input_path),
                    "module3_summary_artifact": str(summary_path),
                },
            )
            context.iteration_records.append(record)
            context.dataset = dataset
            context.emulator = emulator
            context.metadata.update(
                {
                    "status": "running",
                    "completed_iterations": int(len(context.iteration_records)),
                    "pending_iteration": (
                        int(iteration_index + 1)
                        if iteration_index < int(self.config.sampling.iterations)
                        else None
                    ),
                }
            )
            self._save_iteration_progress(context, record)

        validation_dir = run_results_subdir(run_dir, "active_learning_validation", create=True)
        validation = evaluate_emulator_on_validation_set(
            self.config,
            self.camb_data_provider,
            emulator,
            validation_dir,
            asset_version="active_learning_validation",
            progress_callback=self._emit_progress,
            metadata={
                "mode": "active_learning_validation",
                "train_size": int(dataset.raw_thetas.shape[0]),
            },
            validation_bank=self.validation_bank,
            force_rebuild_validation_cache=self.force_rebuild_cache,
        )
        context.validation = validation

        if self.config.gp_baseline.enabled:
            run_standard_sobol_gp_baseline(
                run_dir=run_dir,
                config=self.config,
                test_set_results_path=validation.test_set_results_path,
                spectrum_type=str(resolve_data_source(self.config).spectrum_type),
                camb_data_provider=self.camb_data_provider,
                output_subdir=self.config.gp_baseline.output_subdir,
                n_train_points=int(self.config.gp_baseline.train_points),
                progress_callback=self._emit_progress,
                force_rebuild_training_cache=self.force_rebuild_cache,
            )

        for name, hook in self.extension_hooks.items():
            output = hook(context)
            context.extension_outputs[name] = dict(output or {})

        context.metadata.update(
            {
                "status": "completed",
                "completed_iterations": int(len(context.iteration_records)),
                "final_train_size": int(dataset.raw_thetas.shape[0]),
                "validation_results": str(validation.test_set_results_path),
                "registered_extensions": sorted(self.extension_hooks.keys()),
            }
        )
        return self._complete_run_with_dynamic_reset(context, dynamic_initial_state)
