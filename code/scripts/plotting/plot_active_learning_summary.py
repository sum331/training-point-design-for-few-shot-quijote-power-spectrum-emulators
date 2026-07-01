"""Plot active-learning validation and iteration summaries from one run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.plotting.plot_corner import plot_corner_distribution
from scripts.plotting.plot_ratio_centerline import plot_ratio_centerline_from_test_set
from scripts.plotting.plot_ratio_evolution import plot_ratio_evolution_from_test_set
from scripts.plotting._z2_plot_io import load_payload
from z2quijote.runtime_core.run_artifacts import run_process_path, run_results_path

_DESIGN_LABELS = {
    "ppr32_plus_z2_active32": "PPR32 + Z2 AL32",
    "sobol64": "Sobol64",
    "ppr32": "PPR32",
    "ppr32_plus_sobol32": "PPR32 + Sobol32",
}


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _extract_train_points(payload: dict) -> int | None:
    metadata = dict(payload.get("metadata", {}))
    raw_value = (
        metadata.get("train_points")
        or metadata.get("train_size")
        or payload.get("train_points")
        or payload.get("train_size")
    )
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _build_series_label(prefix: str, metric: str, train_points: int | None) -> str:
    if train_points is not None:
        return f"{prefix} ({train_points} pts) - {metric}"
    return f"{prefix} - {metric}"


def _payload_design_label(payload: dict, fallback: str) -> str:
    metadata = dict(payload.get("metadata", {}))
    raw_name = (
        metadata.get("design_label")
        or metadata.get("design_name")
        or payload.get("design_label")
        or payload.get("design_name")
    )
    if raw_name is None:
        return fallback
    name = str(raw_name)
    return _DESIGN_LABELS.get(name, name)


def _load_validation_series(results_path: Path) -> dict:
    payload = load_payload(results_path)
    return {
        "path": Path(results_path).resolve(),
        "payload": payload,
        "train_points": _extract_train_points(payload),
        "label_prefix": _payload_design_label(payload, "Validation"),
        "k_bins": np.asarray(payload["k_bins"], dtype=np.float64),
        "rel_p68": np.asarray(payload.get("power_relative_error_p68", []), dtype=np.float64),
        "rel_mean": np.asarray(payload.get("power_relative_error_mean", []), dtype=np.float64),
        "log_p68": np.asarray(payload.get("power_log_error_p68", []), dtype=np.float64),
        "log_mean": np.asarray(payload.get("power_log_error_mean", []), dtype=np.float64),
    }


def _load_optional_fixed_budget_validation(run_dir: Path) -> dict | None:
    fixed_budget_path = run_results_path(
        run_dir,
        "fixed_budget_comparison",
        "test_set_results.json",
    )
    if not fixed_budget_path.exists():
        return None

    return _load_validation_series(fixed_budget_path)


def _metric_value(payload: dict, key: str) -> float:
    raw_value = payload.get(key, 0.0)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def _metric_value_with_fallback(payload: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in payload:
            return _metric_value(payload, key)
    return 0.0


def _expanded_segment_metrics_available(payload: dict) -> bool:
    return any(
        key in payload
        for key in (
            "band_relative_error_focus_high_mean",
            "band_relative_error_tail_mean",
            "focus_0p1_5_integrated_relative_error_mean",
        )
    )


def _segment_metric_specs(*, use_expanded: bool) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    if use_expanded:
        return [
            ("Overall", ("overall_mean_relative_error",), ("overall_p68_relative_error",)),
            (r"$k \leq 1$", ("k_le_1_mean_relative_error",), ("k_le_1_p68_relative_error",)),
            ("Low", ("band_relative_error_low_mean",), ("band_relative_error_low_p68",)),
            ("Mid", ("band_relative_error_mid_mean",), ("band_relative_error_mid_p68",)),
            ("1~5", ("band_relative_error_focus_high_mean",), ("band_relative_error_focus_high_p68",)),
            ("5~10", ("band_relative_error_tail_mean",), ("band_relative_error_tail_p68",)),
            (
                "0.1~5 int",
                ("focus_0p1_5_integrated_relative_error_mean",),
                ("focus_0p1_5_integrated_relative_error_p68",),
            ),
        ]
    return [
        ("Overall", ("overall_mean_relative_error",), ("overall_p68_relative_error",)),
        (r"$k \leq 1$", ("k_le_1_mean_relative_error",), ("k_le_1_p68_relative_error",)),
        ("Low", ("band_relative_error_low_mean",), ("band_relative_error_low_p68",)),
        ("Mid", ("band_relative_error_mid_mean",), ("band_relative_error_mid_p68",)),
        ("High", ("band_relative_error_high_mean",), ("band_relative_error_high_p68",)),
    ]


def _segment_metric_arrays(
    payload: dict,
    *,
    use_expanded: bool | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if use_expanded is None:
        use_expanded = _expanded_segment_metrics_available(payload)
    specs = _segment_metric_specs(use_expanded=bool(use_expanded))
    labels = [label for label, _, _ in specs]
    mean_values = np.asarray(
        [_metric_value_with_fallback(payload, mean_keys) for _, mean_keys, _ in specs],
        dtype=np.float64,
    )
    p68_values = np.asarray(
        [_metric_value_with_fallback(payload, p68_keys) for _, _, p68_keys in specs],
        dtype=np.float64,
    )
    return labels, mean_values, p68_values


def _percent_improvement(active_values: np.ndarray, fixed_values: np.ndarray) -> np.ndarray:
    safe_fixed = np.asarray(fixed_values, dtype=np.float64)
    active_arr = np.asarray(active_values, dtype=np.float64)
    improvement = np.zeros_like(active_arr, dtype=np.float64)
    mask = np.abs(safe_fixed) > 1.0e-30
    improvement[mask] = (safe_fixed[mask] - active_arr[mask]) / safe_fixed[mask] * 100.0
    return improvement


def _annotate_segment_improvement(
    ax: plt.Axes,
    x_positions: np.ndarray,
    active_values: np.ndarray,
    fixed_values: np.ndarray,
    deltas_pct: np.ndarray,
) -> float:
    max_values = np.maximum(active_values, fixed_values)
    y_floor = float(np.max(max_values)) if max_values.size > 0 else 0.0
    if y_floor > 0.0:
        y_padding = max(y_floor * 0.08, 1.0e-18)
    else:
        y_padding = 0.05
    y_top = 0.0
    for idx, delta in enumerate(deltas_pct):
        color = "tab:green" if delta >= 0.0 else "tab:red"
        y_pos = float(max_values[idx] + y_padding)
        y_top = max(y_top, y_pos)
        ax.text(
            float(x_positions[idx]),
            y_pos,
            f"{delta:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
    return y_top


def _plot_segment_error_comparison(
    *,
    output_dir: Path,
    active_payload: dict,
    active_train_points: int | None,
    fixed_payload: dict,
    fixed_train_points: int | None,
    active_label_prefix: str = "Active learning",
    fixed_label_prefix: str = "Fixed-budget Sobol-GP",
    title: str = "Segmented relative-error comparison",
    footer_text: str = "Positive annotation means active learning is better than fixed-budget; negative means worse.",
    output_filename: str = "active_vs_fixed_budget_segment_error_comparison.png",
) -> Path:
    use_expanded = _expanded_segment_metrics_available(active_payload) or _expanded_segment_metrics_available(
        fixed_payload
    )
    labels, active_mean, active_p68 = _segment_metric_arrays(active_payload, use_expanded=use_expanded)
    _, fixed_mean, fixed_p68 = _segment_metric_arrays(fixed_payload, use_expanded=use_expanded)
    mean_improvement = _percent_improvement(active_mean, fixed_mean)
    p68_improvement = _percent_improvement(active_p68, fixed_p68)

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    fig, (ax_mean, ax_p68) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_mean.bar(
        x - width / 2.0,
        active_mean,
        width=width,
        color="tab:blue",
        alpha=0.9,
        label=(
            f"{active_label_prefix} ({active_train_points} pts)"
            if active_train_points is not None
            else active_label_prefix
        ),
    )
    ax_mean.bar(
        x + width / 2.0,
        fixed_mean,
        width=width,
        color="tab:green",
        alpha=0.8,
        label=(
            f"{fixed_label_prefix} ({fixed_train_points} pts)"
            if fixed_train_points is not None
            else fixed_label_prefix
        ),
    )
    mean_top = _annotate_segment_improvement(ax_mean, x, active_mean, fixed_mean, mean_improvement)
    if mean_top > 0.0:
        ax_mean.set_ylim(0.0, mean_top * 1.18)
    ax_mean.set_ylabel("Mean relative error")
    ax_mean.set_title(title)
    ax_mean.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_mean.legend()

    ax_p68.bar(
        x - width / 2.0,
        active_p68,
        width=width,
        color="tab:orange",
        alpha=0.9,
        label=(
            f"{active_label_prefix} ({active_train_points} pts)"
            if active_train_points is not None
            else active_label_prefix
        ),
    )
    ax_p68.bar(
        x + width / 2.0,
        fixed_p68,
        width=width,
        color="tab:red",
        alpha=0.8,
        label=(
            f"{fixed_label_prefix} ({fixed_train_points} pts)"
            if fixed_train_points is not None
            else fixed_label_prefix
        ),
    )
    p68_top = _annotate_segment_improvement(ax_p68, x, active_p68, fixed_p68, p68_improvement)
    if p68_top > 0.0:
        ax_p68.set_ylim(0.0, p68_top * 1.18)
    ax_p68.set_ylabel("P68 relative error")
    ax_p68.set_xlabel("Validation segment")
    ax_p68.set_xticks(x)
    ax_p68.set_xticklabels(labels)
    ax_p68.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_p68.legend()

    fig.text(
        0.5,
        0.01,
        footer_text,
        ha="center",
        fontsize=9,
    )
    output_path = output_dir / output_filename
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_validation_error_figure(
    *,
    output_dir: Path,
    active_series: dict,
    active_label_prefix: str,
    active_title_prefix: str,
    output_filename: str,
    ylabel: str,
    active_mean_key: str,
    active_p68_key: str,
    fixed_series: dict | None = None,
    fixed_label_prefix: str = "Fixed-budget Sobol-GP",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        active_series["k_bins"],
        active_series[active_mean_key],
        label=_build_series_label(active_label_prefix, f"Mean {active_title_prefix}", active_series["train_points"]),
        linewidth=1.8,
        color="tab:blue",
    )
    ax.plot(
        active_series["k_bins"],
        active_series[active_p68_key],
        label=_build_series_label(active_label_prefix, f"P68 {active_title_prefix}", active_series["train_points"]),
        linewidth=1.4,
        color="tab:orange",
    )
    if fixed_series is not None and fixed_series[active_mean_key].size > 0 and fixed_series[active_p68_key].size > 0:
        ax.plot(
            fixed_series["k_bins"],
            fixed_series[active_mean_key],
            label=_build_series_label(fixed_label_prefix, f"Mean {active_title_prefix}", fixed_series["train_points"]),
            linewidth=1.8,
            linestyle="--",
            color="tab:green",
        )
        ax.plot(
            fixed_series["k_bins"],
            fixed_series[active_p68_key],
            label=_build_series_label(fixed_label_prefix, f"P68 {active_title_prefix}", fixed_series["train_points"]),
            linewidth=1.4,
            linestyle="--",
            color="tab:red",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel(ylabel)
    if fixed_series is not None:
        ax.set_title(f"{active_label_prefix} vs {fixed_label_prefix} emulator validation" + (" (log error)" if "log" in active_title_prefix else ""))
    else:
        ax.set_title(f"{active_label_prefix} emulator validation" + (" (log error)" if "log" in active_title_prefix else ""))
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend()
    output_path = output_dir / output_filename
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_custom_validation_comparison(
    *,
    active_results_path: Path,
    fixed_results_path: Path,
    output_dir: Path,
    active_label_prefix: str,
    fixed_label_prefix: str,
    output_prefix: str,
    segment_title: str,
    footer_text: str,
) -> list[Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    active_series = _load_validation_series(Path(active_results_path).resolve())
    fixed_series = _load_validation_series(Path(fixed_results_path).resolve())

    outputs = [
        _plot_validation_error_figure(
            output_dir=output_dir,
            active_series=active_series,
            active_label_prefix=active_label_prefix,
            active_title_prefix="relative error",
            output_filename=f"{output_prefix}_validation_error.png",
            ylabel="Relative error",
            active_mean_key="rel_mean",
            active_p68_key="rel_p68",
            fixed_series=fixed_series,
            fixed_label_prefix=fixed_label_prefix,
        ),
        _plot_validation_error_figure(
            output_dir=output_dir,
            active_series=active_series,
            active_label_prefix=active_label_prefix,
            active_title_prefix="log error",
            output_filename=f"{output_prefix}_validation_log_error.png",
            ylabel="Absolute log error",
            active_mean_key="log_mean",
            active_p68_key="log_p68",
            fixed_series=fixed_series,
            fixed_label_prefix=fixed_label_prefix,
        ),
        _plot_segment_error_comparison(
            output_dir=output_dir,
            active_payload=active_series["payload"],
            active_train_points=active_series["train_points"],
            fixed_payload=fixed_series["payload"],
            fixed_train_points=fixed_series["train_points"],
            active_label_prefix=active_label_prefix,
            fixed_label_prefix=fixed_label_prefix,
            title=segment_title,
            footer_text=footer_text,
            output_filename=f"{output_prefix}_segment_error_comparison.png",
        ),
    ]
    return outputs


def _build_module3_score_matrix(history: list[dict]) -> tuple[np.ndarray, list[int]]:
    if not history:
        return np.empty((0, 0), dtype=np.float64), []
    max_pc = -1
    for item in history:
        source_pc = np.asarray(item.get("selected_source_pc", []), dtype=np.int64).reshape(-1)
        if source_pc.size > 0:
            max_pc = max(max_pc, int(np.max(source_pc)))
        component_count = int(item.get("metadata", {}).get("component_count", 0))
        if component_count > 0:
            max_pc = max(max_pc, component_count - 1)
    if max_pc < 0:
        return np.empty((len(history), 0), dtype=np.float64), [int(item["iteration_index"]) for item in history]

    matrix = np.full((len(history), max_pc + 1), np.nan, dtype=np.float64)
    iteration_ids = [int(item["iteration_index"]) for item in history]
    for row_idx, item in enumerate(history):
        source_pc = np.asarray(item.get("selected_source_pc", []), dtype=np.int64).reshape(-1)
        scores = np.asarray(item.get("selected_scores", []), dtype=np.float64).reshape(-1)
        for pc_idx, score in zip(source_pc, scores, strict=True):
            if pc_idx >= 0:
                matrix[row_idx, int(pc_idx)] = float(score)
    return matrix, iteration_ids


def _plot_module3_diagnostics(
    history: list[dict],
    output_dir: Path,
) -> list[Path]:
    if not history:
        return []

    outputs: list[Path] = []
    score_matrix, iteration_ids = _build_module3_score_matrix(history)
    if score_matrix.size > 0:
        fig_heatmap, ax_heatmap = plt.subplots(
            figsize=(max(8.0, 0.45 * score_matrix.shape[1] + 2.5), max(3.6, 0.7 * score_matrix.shape[0] + 2.0))
        )
        masked = np.ma.masked_invalid(score_matrix)
        image = ax_heatmap.imshow(masked, aspect="auto", interpolation="nearest", cmap="viridis")
        ax_heatmap.set_xlabel("PCA component index")
        ax_heatmap.set_ylabel("Iteration")
        ax_heatmap.set_title("Module3 selected posterior variance by PCA component")
        ax_heatmap.set_xticks(np.arange(score_matrix.shape[1]))
        ax_heatmap.set_yticks(np.arange(len(iteration_ids)))
        ax_heatmap.set_yticklabels([str(item) for item in iteration_ids])
        cbar = fig_heatmap.colorbar(image, ax=ax_heatmap)
        cbar.set_label("Selected score")
        path_heatmap = output_dir / "module3_selected_scores_heatmap.png"
        fig_heatmap.savefig(path_heatmap, dpi=250, bbox_inches="tight")
        plt.close(fig_heatmap)
        outputs.append(path_heatmap)

    score_mean = []
    score_min = []
    score_max = []
    num_simplices = []
    for item in history:
        scores = np.asarray(item.get("selected_scores", []), dtype=np.float64).reshape(-1)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size > 0:
            score_mean.append(float(np.mean(finite_scores)))
            score_min.append(float(np.min(finite_scores)))
            score_max.append(float(np.max(finite_scores)))
        else:
            score_mean.append(0.0)
            score_min.append(0.0)
            score_max.append(0.0)
        num_simplices.append(int(item.get("metadata", {}).get("num_simplices", 0)))

    fig_diag, (ax_score, ax_geom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax_score.plot(iteration_ids, score_mean, marker="o", linewidth=1.8, label="Mean selected score")
    ax_score.plot(iteration_ids, score_max, marker="^", linewidth=1.4, label="Max selected score")
    ax_score.plot(iteration_ids, score_min, marker="v", linewidth=1.2, label="Min selected score")
    ax_score.set_ylabel("Selected score")
    ax_score.set_title("Module3 score summary by iteration")
    ax_score.grid(True, linestyle="--", alpha=0.25)
    ax_score.legend()

    ax_geom.plot(iteration_ids, num_simplices, marker="s", linewidth=1.6, color="tab:orange")
    ax_geom.set_xlabel("Iteration")
    ax_geom.set_ylabel("Shared hull simplices")
    ax_geom.set_title("Module3 shared Delaunay geometry")
    ax_geom.grid(True, linestyle="--", alpha=0.25)

    path_diag = output_dir / "module3_iteration_diagnostics.png"
    fig_diag.savefig(path_diag, dpi=250, bbox_inches="tight")
    plt.close(fig_diag)
    outputs.append(path_diag)
    return outputs


def plot_active_learning_summary(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    run_dir = Path(run_dir).resolve()
    output_dir = (Path(output_dir).resolve() if output_dir is not None else run_results_path(run_dir, "plots", create=True))
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = run_results_path(
        run_dir,
        "active_learning_validation",
        "test_set_results.json",
    )
    iteration_path = run_process_path(run_dir, "iteration_history.json")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation results not found: {validation_path}")

    validation = load_payload(validation_path)
    active_train_points = _extract_train_points(validation)
    active_label_prefix = _payload_design_label(validation, "Active learning")
    k_bins = np.asarray(validation["k_bins"], dtype=np.float64)
    rel_p68 = np.asarray(validation["power_relative_error_p68"], dtype=np.float64)
    rel_mean = np.asarray(validation["power_relative_error_mean"], dtype=np.float64)
    log_p68 = np.asarray(validation.get("power_log_error_p68", []), dtype=np.float64)
    log_mean = np.asarray(validation.get("power_log_error_mean", []), dtype=np.float64)
    sample_mean_rel = np.asarray(validation.get("sample_mean_relative_error", []), dtype=np.float64)
    sample_max_rel = np.asarray(validation.get("sample_max_relative_error", []), dtype=np.float64)
    sample_mean_log = np.asarray(validation.get("sample_mean_log_error", []), dtype=np.float64)
    sample_max_log = np.asarray(validation.get("sample_max_log_error", []), dtype=np.float64)
    fixed_budget = _load_optional_fixed_budget_validation(run_dir)

    outputs: list[Path] = []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        k_bins,
        rel_mean,
        label=_build_series_label(active_label_prefix, "Mean relative error", active_train_points),
        linewidth=1.8,
        color="tab:blue",
    )
    ax.plot(
        k_bins,
        rel_p68,
        label=_build_series_label(active_label_prefix, "P68 relative error", active_train_points),
        linewidth=1.4,
        color="tab:orange",
    )
    if fixed_budget is not None and fixed_budget["rel_mean"].size > 0 and fixed_budget["rel_p68"].size > 0:
        ax.plot(
            fixed_budget["k_bins"],
            fixed_budget["rel_mean"],
            label=_build_series_label(
                fixed_budget["label_prefix"],
                "Mean relative error",
                fixed_budget["train_points"],
            ),
            linewidth=1.8,
            linestyle="--",
            color="tab:green",
        )
        ax.plot(
            fixed_budget["k_bins"],
            fixed_budget["rel_p68"],
            label=_build_series_label(
                fixed_budget["label_prefix"],
                "P68 relative error",
                fixed_budget["train_points"],
            ),
            linewidth=1.4,
            linestyle="--",
            color="tab:red",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel("Relative error")
    if fixed_budget is not None:
        ax.set_title(f"{active_label_prefix} vs {fixed_budget['label_prefix']} emulator validation")
    else:
        ax.set_title(f"{active_label_prefix} emulator validation")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend()
    path_error = output_dir / "active_learning_validation_error.png"
    fig.savefig(path_error, dpi=250, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path_error)

    if log_p68.size > 0:
        fig_log, ax_log = plt.subplots(figsize=(9, 5))
        ax_log.plot(
            k_bins,
            log_p68,
            label=_build_series_label(active_label_prefix, "P68 log error", active_train_points),
            linewidth=1.8,
            color="tab:orange",
        )
        if fixed_budget is not None and fixed_budget["log_p68"].size > 0:
            ax_log.plot(
                fixed_budget["k_bins"],
                fixed_budget["log_p68"],
                label=_build_series_label(
                    fixed_budget["label_prefix"],
                    "P68 log error",
                    fixed_budget["train_points"],
                ),
                linewidth=1.4,
                linestyle="--",
                color="tab:red",
            )
        ax_log.set_xscale("log")
        ax_log.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        ax_log.set_ylabel("Absolute log error")
        if fixed_budget is not None:
            ax_log.set_title(f"{active_label_prefix} vs {fixed_budget['label_prefix']} emulator validation (log error)")
        else:
            ax_log.set_title(f"{active_label_prefix} emulator validation (log error)")
        ax_log.grid(True, which="both", linestyle="--", alpha=0.25)
        ax_log.legend()
        path_log = output_dir / "active_learning_validation_log_error.png"
        fig_log.savefig(path_log, dpi=250, bbox_inches="tight")
        plt.close(fig_log)
        outputs.append(path_log)

    if fixed_budget is not None:
        outputs.append(
            _plot_segment_error_comparison(
                output_dir=output_dir,
                active_payload=validation,
                active_train_points=active_train_points,
                fixed_payload=fixed_budget["payload"],
                fixed_train_points=fixed_budget["train_points"],
                active_label_prefix=active_label_prefix,
                fixed_label_prefix=fixed_budget["label_prefix"],
                title=f"{active_label_prefix} vs {fixed_budget['label_prefix']} segmented relative-error comparison",
            )
        )

    if sample_mean_rel.size > 0 and sample_max_rel.size > 0:
        order = np.argsort(sample_max_rel)
        sample_index = np.arange(1, sample_max_rel.shape[0] + 1)
        fig_sample, ax_sample = plt.subplots(figsize=(9, 5))
        ax_sample.plot(sample_index, sample_mean_rel[order], label="Sample mean relative error", linewidth=1.6)
        ax_sample.plot(sample_index, sample_max_rel[order], label="Sample max relative error", linewidth=1.4)
        ax_sample.set_xlabel("Sorted validation sample index")
        ax_sample.set_ylabel("Relative error")
        ax_sample.set_title(f"{active_label_prefix} sample-level relative error")
        ax_sample.grid(True, linestyle="--", alpha=0.25)
        ax_sample.legend()
        path_sample = output_dir / "active_learning_sample_relative_error.png"
        fig_sample.savefig(path_sample, dpi=250, bbox_inches="tight")
        plt.close(fig_sample)
        outputs.append(path_sample)

    if sample_mean_log.size > 0 and sample_max_log.size > 0:
        order_log = np.argsort(sample_max_log)
        sample_index_log = np.arange(1, sample_max_log.shape[0] + 1)
        fig_sample_log, ax_sample_log = plt.subplots(figsize=(9, 5))
        ax_sample_log.plot(sample_index_log, sample_mean_log[order_log], label="Sample mean log error", linewidth=1.6)
        ax_sample_log.plot(sample_index_log, sample_max_log[order_log], label="Sample max log error", linewidth=1.4)
        ax_sample_log.set_xlabel("Sorted validation sample index")
        ax_sample_log.set_ylabel("Absolute log error")
        ax_sample_log.set_title(f"{active_label_prefix} sample-level log error")
        ax_sample_log.grid(True, linestyle="--", alpha=0.25)
        ax_sample_log.legend()
        path_sample_log = output_dir / "active_learning_sample_log_error.png"
        fig_sample_log.savefig(path_sample_log, dpi=250, bbox_inches="tight")
        plt.close(fig_sample_log)
        outputs.append(path_sample_log)

    if iteration_path.exists():
        history = _load_json(iteration_path)
        if history:
            xs = [item["iteration_index"] for item in history]
            ys = [item["train_size_after"] for item in history]
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(xs, ys, marker="o", linewidth=1.8)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Training set size")
            ax2.set_title("Active-learning sample growth")
            ax2.grid(True, linestyle="--", alpha=0.25)
            path_growth = output_dir / "active_learning_train_size.png"
            fig2.savefig(path_growth, dpi=250, bbox_inches="tight")
            plt.close(fig2)
            outputs.append(path_growth)
            outputs.extend(_plot_module3_diagnostics(history, output_dir))

    outputs.extend(
        plot_ratio_evolution_from_test_set(
            validation_path,
            output_dir / "ratio_evolution.png",
        )
    )
    outputs.extend(
        plot_ratio_centerline_from_test_set(
            validation_path,
            output_dir / "ratio_centerline.png",
            title_label=active_label_prefix,
        )
    )
    if fixed_budget is not None:
        fixed_budget_path = run_results_path(
            run_dir,
            "fixed_budget_comparison",
            "test_set_results.json",
        )
        outputs.extend(
            plot_ratio_centerline_from_test_set(
                fixed_budget_path,
                output_dir / "fixed_budget_ratio_centerline.png",
                title_label=fixed_budget["label_prefix"],
            )
        )
    outputs.append(
        plot_corner_distribution(
            run_dir,
            output_dir / "corner_distribution.png",
        )
    )

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot active-learning summary figures for one run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    outputs = plot_active_learning_summary(args.run_dir, args.output_dir)
    for path in outputs:
        print(f"[Plot] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
