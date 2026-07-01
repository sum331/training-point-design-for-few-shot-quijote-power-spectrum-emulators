"""Sweep-specific plotting helpers for budget convergence and per-run diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from z2quijote.runtime_core.config import ValidationRuntimeConfig, load_config
from z2quijote.runtime_core.data_source import resolve_data_source
from z2quijote.runtime_core.evaluation.test_set import build_test_set_results_payload
from z2quijote.runtime_core.module1_facade import build_dataset_from_spectrum_bank
from z2quijote.runtime_core.module2_facade import fit_emulator, predict_spectra
from z2quijote.runtime_core.run_artifacts import run_process_path


_BAND_LABELS = ["0.01-0.1", "0.1-1", "1-5", "5-10"]
_PARAMETER_LABELS: dict[str, str] = {
    "Omegab": r"$\Omega_\mathrm{b}$",
    "Omegacb": r"$\Omega_\mathrm{cb}$",
    "Omegam": r"$\Omega_\mathrm{m}$",
    "Omega_m": r"$\Omega_\mathrm{m}$",
    "Omega_b": r"$\Omega_\mathrm{b}$",
    "H0": r"$H_0$",
    "h": r"$h$",
    "ns": r"$n_\mathrm{s}$",
    "n_s": r"$n_\mathrm{s}$",
    "A": r"$10^9 A_\mathrm{s}$",
    "sigma_8": r"$\sigma_8$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
    "mnu": r"$\Sigma m_\nu$",
}
_TRACE_CACHE_VERSION = 1
_ACTIVE_COLOR = "#0072B2"
_BASELINE_COLOR = "#D55E00"
_FOCUS_COLOR = "#009E73"
_LOWK_COLOR = "#6A3D9A"
_GRID_ALPHA = 0.22
_ACTIVE_CMAP_NAME = "Blues"
_BASELINE_CMAP_NAME = "Oranges"


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _records_from_summary(summary_json_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(Path(summary_json_path).resolve())
    records = list(payload.get("records", []))
    return sorted(records, key=lambda item: int(item["budget"]))


def _load_results_payload(results_path: Path) -> dict[str, Any]:
    return _load_json(Path(results_path).resolve())


def _style_axis(ax: plt.Axes, *, logy: bool = False, logx: bool = False) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=_GRID_ALPHA)
    ax.grid(True, which="minor", linestyle=":", alpha=_GRID_ALPHA * 0.65)
    ax.tick_params(labelsize=10)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )


def _budget_ticks(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(item["budget"]) for item in records], dtype=np.int64)


def _band_metric_arrays(metrics: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(metrics.get("band_relative_error_low_mean", 0.0)),
            float(metrics.get("band_relative_error_mid_mean", 0.0)),
            float(metrics.get("band_relative_error_focus_high_mean", 0.0)),
            float(metrics.get("band_relative_error_tail_mean", 0.0)),
        ],
        dtype=np.float64,
    )


def _metric_triptych_specs() -> list[tuple[str, str, str]]:
    return [
        ("overall_mean_relative_error", "Overall", "Overall mean relative error"),
        (
            "focus_0p1_5_integrated_relative_error_mean",
            "0.1~5",
            r"Integrated relative error ($0.1 \leq k \leq 5$)",
        ),
        ("k_le_1_mean_relative_error", "k<=1", r"Mean relative error ($k \leq 1$)"),
    ]


def _budget_metric_panel_specs() -> list[tuple[str, str, str]]:
    return [
        ("overall_mean_relative_error", "Overall", "Overall mean relative error"),
        ("band_relative_error_low_mean", r"$k \in [0.01, 0.1]$", "Mean relative error"),
        ("band_relative_error_mid_mean", r"$k \in [0.1, 1]$", "Mean relative error"),
        ("band_relative_error_focus_high_mean", r"$k \in [1, 5]$", "Mean relative error"),
        ("band_relative_error_tail_mean", r"$k \in [5, 10]$", "Mean relative error"),
    ]


def plot_budget_metric_triptych(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    budgets = _budget_ticks(records)
    specs = _budget_metric_panel_specs()

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.2), sharex=True)
    axes_flat = list(axes.flat)
    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, (metric_key, short_label, ylabel), panel_tag in zip(
        axes_flat[: len(specs)], specs, panel_tags, strict=True
    ):
        active = np.asarray(
            [float(item["active_metrics"].get(metric_key, np.nan)) for item in records],
            dtype=np.float64,
        )
        fixed = np.asarray(
            [float(item["baseline_metrics"].get(metric_key, np.nan)) for item in records],
            dtype=np.float64,
        )
        line_active = ax.plot(
            budgets,
            active,
            marker="o",
            linewidth=2.2,
            color=_ACTIVE_COLOR,
            label="Active learning",
        )[0]
        line_fixed = ax.plot(
            budgets,
            fixed,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color=_BASELINE_COLOR,
            label="Traditional GP",
        )[0]
        _style_axis(ax, logy=True)
        ax.set_ylabel(ylabel)
        ax.set_title(short_label, fontsize=12)
        ax.set_xticks(budgets)
        _panel_label(ax, panel_tag)

    axes_flat[len(specs)].axis("off")
    axes[2, 0].set_xlabel("Total training budget")
    axes[2, 1].set_xlabel("Total training budget")
    from matplotlib.lines import Line2D

    fig.legend(
        [
            Line2D([0], [0], color=_ACTIVE_COLOR, linewidth=2.2, marker="o"),
            Line2D([0], [0], color=_BASELINE_COLOR, linewidth=2.0, linestyle="--", marker="s"),
        ],
        ["Active learning", "Traditional GP"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.972),
    )
    fig.suptitle("Validation-error scaling with total training budget", fontsize=15, y=0.992)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_budget_gain_triptych(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    budgets = _budget_ticks(records)
    specs = _budget_metric_panel_specs()

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.2), sharex=True)
    axes_flat = list(axes.flat)
    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, (metric_key, short_label, _), panel_tag in zip(
        axes_flat[: len(specs)], specs, panel_tags, strict=True
    ):
        active = np.asarray(
            [float(item["active_metrics"].get(metric_key, np.nan)) for item in records],
            dtype=np.float64,
        )
        fixed = np.asarray(
            [float(item["baseline_metrics"].get(metric_key, np.nan)) for item in records],
            dtype=np.float64,
        )
        gain_pct = np.zeros_like(active, dtype=np.float64)
        mask = np.abs(fixed) > 1.0e-30
        gain_pct[mask] = (fixed[mask] - active[mask]) / fixed[mask] * 100.0
        colors = [_FOCUS_COLOR if item >= 0.0 else "#C7362F" for item in gain_pct]
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.plot(budgets, gain_pct, color="#666666", linewidth=1.1, alpha=0.7, zorder=1)
        ax.vlines(budgets, 0.0, gain_pct, color=colors, linewidth=1.6, alpha=0.9, zorder=2)
        ax.scatter(budgets, gain_pct, s=42, c=colors, edgecolors="white", linewidths=0.7, zorder=3)
        for x, y in zip(budgets, gain_pct, strict=True):
            ax.text(
                float(x),
                float(y) + (0.9 if y >= 0.0 else -1.2),
                f"{y:+.1f}",
                fontsize=8.5,
                ha="center",
                va="bottom" if y >= 0.0 else "top",
                color="#333333",
            )
        ax.set_ylabel("Gain vs baseline (%)")
        ax.set_title(short_label, fontsize=12)
        ax.set_xticks(budgets)
        ax.grid(True, axis="y", linestyle="--", alpha=_GRID_ALPHA)
        _panel_label(ax, panel_tag)

    axes_flat[len(specs)].axis("off")
    axes[2, 0].set_xlabel("Total training budget")
    axes[2, 1].set_xlabel("Total training budget")
    fig.suptitle("Relative gain of active learning over traditional GP", fontsize=15, y=0.995)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_budget_band_error_bars(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    budgets = _budget_ticks(records)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True, sharey=False)
    width = 0.36
    x = np.arange(len(records), dtype=np.float64)
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    for ax, band_index, band_label, panel_tag in zip(
        axes.flat,
        range(4),
        _BAND_LABELS,
        panel_tags,
        strict=True,
    ):
        active = np.asarray(
            [_band_metric_arrays(item["active_metrics"])[band_index] for item in records],
            dtype=np.float64,
        )
        fixed = np.asarray(
            [_band_metric_arrays(item["baseline_metrics"])[band_index] for item in records],
            dtype=np.float64,
        )
        bar_active = ax.bar(
            x - width / 2.0,
            active,
            width=width,
            color=_ACTIVE_COLOR,
            alpha=0.88,
            label="Active learning",
        )
        bar_fixed = ax.bar(
            x + width / 2.0,
            fixed,
            width=width,
            color=_BASELINE_COLOR,
            alpha=0.78,
            label="Traditional GP",
        )
        _style_axis(ax, logy=True)
        ax.set_title(rf"$k \in [{band_label}]$", fontsize=12)
        ax.set_ylabel("Mean relative error")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(item)) for item in budgets])
        _panel_label(ax, panel_tag)

    axes[-1, 0].set_xlabel("Total training budget")
    axes[-1, 1].set_xlabel("Total training budget")
    from matplotlib.lines import Line2D

    fig.legend(
        [
            Line2D([0], [0], color=_ACTIVE_COLOR, linewidth=7),
            Line2D([0], [0], color=_BASELINE_COLOR, linewidth=7),
        ],
        ["Active learning", "Traditional GP"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.972),
    )
    fig.suptitle("Bandwise validation error across training budgets", fontsize=15, y=0.992)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_budget_p68_gradient(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    n = max(len(records), 1)
    active_cmap = matplotlib.colormaps[_ACTIVE_CMAP_NAME]
    fixed_cmap = matplotlib.colormaps[_BASELINE_CMAP_NAME]
    active_colors = active_cmap(np.linspace(0.45, 0.95, n))
    fixed_colors = fixed_cmap(np.linspace(0.45, 0.95, n))
    budgets = _budget_ticks(records).astype(np.float64)
    budget_norm = mcolors.Normalize(vmin=float(np.min(budgets)), vmax=float(np.max(budgets)))

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    for idx, item in enumerate(records):
        budget = int(item["budget"])
        active_payload = _load_results_payload(Path(item["active_results_path"]))
        fixed_payload = _load_results_payload(Path(item["baseline_results_path"]))
        active_k = np.asarray(active_payload["k_bins"], dtype=np.float64)
        active_p68 = np.asarray(active_payload["power_relative_error_p68"], dtype=np.float64)
        fixed_k = np.asarray(fixed_payload["k_bins"], dtype=np.float64)
        fixed_p68 = np.asarray(fixed_payload["power_relative_error_p68"], dtype=np.float64)

        ax.plot(
            active_k,
            active_p68,
            color=active_colors[idx],
            linewidth=2.0,
            alpha=0.95,
        )
        ax.plot(
            fixed_k,
            fixed_p68,
            color=fixed_colors[idx],
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        )

    for boundary in (0.1, 1.0, 5.0):
        ax.axvline(boundary, color="#888888", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.axhline(0.01, color="#B22222", linestyle="--", linewidth=1.1)
    _style_axis(ax, logx=True, logy=True)
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel("P68 relative error")
    ax.set_title("P68 error envelopes across budgets", fontsize=14)

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="-", label="Active learning"),
        Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="--", label="Traditional GP"),
        Line2D([0], [0], color="#B22222", linewidth=1.1, linestyle="--", label="Target 1e-2"),
    ]
    method_legend = ax.legend(
        handles=method_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
    )
    ax.add_artist(method_legend)

    sm_active = cm.ScalarMappable(norm=budget_norm, cmap=active_cmap)
    sm_active.set_array([])
    sm_fixed = cm.ScalarMappable(norm=budget_norm, cmap=fixed_cmap)
    sm_fixed.set_array([])
    cax_active = fig.add_axes([0.88, 0.57, 0.018, 0.24])
    cax_fixed = fig.add_axes([0.88, 0.18, 0.018, 0.24])
    cbar_active = fig.colorbar(sm_active, cax=cax_active)
    cbar_fixed = fig.colorbar(sm_fixed, cax=cax_fixed)
    cbar_active.set_label("Active budget", fontsize=9)
    cbar_fixed.set_label("GP budget", fontsize=9)
    cbar_active.set_ticks(budgets)
    cbar_fixed.set_ticks(budgets)
    cbar_active.ax.tick_params(labelsize=8)
    cbar_fixed.ax.tick_params(labelsize=8)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(right=0.84)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _resolve_parameter_labels(training_summary: dict[str, Any], theta_dim: int) -> list[str]:
    parameter_names = list(training_summary.get("parameter_names", []))
    names = list(parameter_names[:theta_dim])
    if len(names) < theta_dim:
        names.extend(f"theta_{idx + 1}" for idx in range(len(names), theta_dim))
    return [_PARAMETER_LABELS.get(name, name) for name in names]


def _match_row_indices(reference_rows: np.ndarray, candidate_rows: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_rows, dtype=np.float64)
    cand = np.asarray(candidate_rows, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"Row arrays must have the same shape, got {ref.shape} vs {cand.shape}.")
    if ref.size > 0 and np.allclose(ref, cand, atol=1.0e-10, rtol=0.0):
        return np.arange(ref.shape[0], dtype=np.int64)

    lookup: dict[tuple[float, ...], list[int]] = {}
    for idx, row in enumerate(cand):
        lookup.setdefault(tuple(np.round(row, 12).tolist()), []).append(idx)

    resolved: list[int] = []
    used: set[int] = set()
    for row in ref:
        key = tuple(np.round(row, 12).tolist())
        match = None
        candidates = lookup.get(key, [])
        while candidates:
            candidate = candidates.pop(0)
            if candidate not in used:
                match = candidate
                break
        if match is None:
            distances = np.max(np.abs(cand - row[None, :]), axis=1)
            order = np.argsort(distances)
            for candidate in order:
                if int(candidate) not in used and float(distances[int(candidate)]) <= 1.0e-8:
                    match = int(candidate)
                    break
        if match is None:
            raise ValueError("Unable to align hifi_bank ordering with training_point_summary.")
        resolved.append(int(match))
        used.add(int(match))
    return np.asarray(resolved, dtype=np.int64)


def _load_ordered_training_bank(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    summary_path = run_process_path(run_dir, "training_point_summary.json")
    bank_path = run_process_path(run_dir, "hifi_bank.npz")
    training_summary = _load_json(summary_path)
    with np.load(bank_path, allow_pickle=False) as npz:
        bank_thetas = np.asarray(npz["train_thetas"], dtype=np.float64)
        bank_k = np.asarray(npz["train_k_bins"], dtype=np.float64)
        bank_nonlin = np.asarray(npz["train_nonlin_pk"], dtype=np.float64)
        bank_linear = None
        if "train_linear_pk" in npz.files:
            bank_linear = np.asarray(npz["train_linear_pk"], dtype=np.float64)

    final_raw = np.asarray(training_summary.get("final_raw_thetas", bank_thetas), dtype=np.float64)
    if final_raw.ndim == 1 and final_raw.size > 0:
        final_raw = final_raw.reshape(1, -1)
    if final_raw.ndim != 2:
        raise ValueError(f"final_raw_thetas must be 2D, got {final_raw.shape}.")

    indices = _match_row_indices(final_raw, bank_thetas)
    ordered_nonlin = bank_nonlin[indices]
    ordered_linear = None if bank_linear is None else bank_linear[indices]
    return final_raw, bank_k, ordered_nonlin, ordered_linear, training_summary


def _trace_cache_payload_matches(cache_payload: dict[str, Any], *, train_sizes: list[int]) -> bool:
    return (
        int(cache_payload.get("trace_cache_version", -1)) == _TRACE_CACHE_VERSION
        and list(cache_payload.get("train_sizes", [])) == list(train_sizes)
    )


def _rebuild_iteration_metric_trace(
    config: ValidationRuntimeConfig,
    *,
    active_results_path: Path,
    run_dir: Path,
    cache_json_path: Path,
) -> dict[str, Any]:
    active_payload = _load_results_payload(active_results_path)
    validation_thetas = np.asarray(active_payload["test_thetas"], dtype=np.float64)
    validation_k = np.asarray(active_payload["k_bins"], dtype=np.float64)
    validation_true = np.asarray(active_payload["p_true_batch"], dtype=np.float64)
    validation_linear = None
    if "p_linear_batch" in active_payload and active_payload["p_linear_batch"] is not None:
        validation_linear = np.asarray(active_payload["p_linear_batch"], dtype=np.float64)

    final_raw, source_k, ordered_nonlin, ordered_linear, training_summary = _load_ordered_training_bank(run_dir)
    iteration_history = _load_json(run_process_path(run_dir, "iteration_history.json"))

    initial_count = int(np.asarray(training_summary.get("initial_raw_thetas", []), dtype=np.float64).shape[0])
    train_sizes = [initial_count]
    current = initial_count
    for item in iteration_history:
        selected = np.asarray(item.get("selected_raw_thetas", []), dtype=np.float64)
        if selected.ndim == 1 and selected.size > 0:
            selected = selected.reshape(1, -1)
        current += int(selected.shape[0]) if selected.ndim == 2 else 0
        train_sizes.append(current)

    if cache_json_path.exists():
        payload = _load_json(cache_json_path)
        if _trace_cache_payload_matches(payload, train_sizes=train_sizes):
            return payload

    metric_rows: list[dict[str, float]] = []
    for train_size in train_sizes:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            dataset = build_dataset_from_spectrum_bank(
                config,
                final_raw[:train_size],
                source_k,
                ordered_nonlin[:train_size],
                p_linear_batch=None if ordered_linear is None else ordered_linear[:train_size],
            )
            emulator = fit_emulator(config, dataset)
            prediction = predict_spectra(
                emulator,
                validation_thetas,
                input_space="raw",
                k_target=validation_k,
                p_linear_batch=validation_linear,
            )
            data_source = resolve_data_source(config)
            results_payload = build_test_set_results_payload(
                validation_thetas,
                validation_k,
                validation_true,
                prediction.pk_mean,
                validation_linear,
                spectrum_type=str(data_source.spectrum_type),
                eps_r=float(config.eps_r),
                metadata={
                    "train_size": int(train_size),
                    "data_source": str(data_source.name),
                    "parameter_space": str(data_source.parameter_space),
                    "theta_dim": int(data_source.theta_dim),
                    "theta_names": list(data_source.theta_names),
                    "target_transform": str(data_source.target_transform),
                },
            )
        metric_rows.append(
            {
                "train_size": int(train_size),
                "overall_mean_relative_error": float(results_payload["overall_mean_relative_error"]),
                "focus_0p1_5_integrated_relative_error_mean": float(
                    results_payload["focus_0p1_5_integrated_relative_error_mean"]
                ),
                "k_le_1_mean_relative_error": float(results_payload["k_le_1_mean_relative_error"]),
            }
        )

    trace_payload = {
        "trace_cache_version": _TRACE_CACHE_VERSION,
        "run_dir": str(Path(run_dir).resolve()),
        "active_results_path": str(Path(active_results_path).resolve()),
        "train_sizes": [int(item) for item in train_sizes],
        "iteration_indices": list(range(len(train_sizes))),
        "metrics": metric_rows,
    }
    _write_json(cache_json_path, trace_payload)
    return trace_payload


def plot_run_iteration_error_decline(
    trace_payload: dict[str, Any],
    output_path: Path,
) -> Path:
    metrics = list(trace_payload.get("metrics", []))
    x = np.asarray(trace_payload.get("iteration_indices", list(range(len(metrics)))), dtype=np.int64)
    train_sizes = np.asarray(trace_payload.get("train_sizes", []), dtype=np.int64)
    if x.size != len(metrics):
        x = np.arange(len(metrics), dtype=np.int64)

    specs = _metric_triptych_specs()
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 9.3), sharex=True)
    panel_tags = ["(a)", "(b)", "(c)"]
    colors = [_ACTIVE_COLOR, _FOCUS_COLOR, _LOWK_COLOR]
    for ax, (metric_key, short_label, ylabel), panel_tag, color in zip(
        axes, specs, panel_tags, colors, strict=True
    ):
        values = np.asarray([float(item.get(metric_key, np.nan)) for item in metrics], dtype=np.float64)
        drop_pct = 0.0
        if values.size > 1 and np.isfinite(values[0]) and abs(values[0]) > 1.0e-30:
            drop_pct = (values[0] - values[-1]) / values[0] * 100.0
        ax.plot(x, values, marker="o", linewidth=2.0, color=color, markersize=4.5)
        _style_axis(ax, logy=True)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{short_label} ({drop_pct:+.1f}% from initial to final)", fontsize=12)
        _panel_label(ax, panel_tag)

    axes[-1].set_xlabel("Iteration (0 = initial design)")
    if train_sizes.size == x.size:
        tick_positions = x
        tick_labels = [f"{int(item)}" for item in train_sizes]
        twin = axes[0].secondary_xaxis("top")
        twin.set_xticks(tick_positions)
        twin.set_xticklabels(tick_labels)
        twin.set_xlabel("Train size")

    fig.suptitle("Reconstructed per-iteration validation trajectory", fontsize=15, y=0.995)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_run_iteration_marginal_gain(
    trace_payload: dict[str, Any],
    output_path: Path,
) -> Path:
    metrics = list(trace_payload.get("metrics", []))
    x_all = np.asarray(trace_payload.get("iteration_indices", list(range(len(metrics)))), dtype=np.int64)
    specs = _metric_triptych_specs()
    if len(metrics) <= 1:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        ax.axis("off")
        ax.text(0.01, 0.5, "Not enough iterations to compute marginal gains.", fontsize=11, va="center")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    x = x_all[1:]
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 9.0), sharex=True)
    panel_tags = ["(a)", "(b)", "(c)"]
    for ax, (metric_key, short_label, _), panel_tag in zip(axes, specs, panel_tags, strict=True):
        values = np.asarray([float(item.get(metric_key, np.nan)) for item in metrics], dtype=np.float64)
        gains = values[:-1] - values[1:]
        colors = [_FOCUS_COLOR if item >= 0.0 else "#C7362F" for item in gains]
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.bar(x, gains, color=colors, alpha=0.88, width=0.8)
        ax.set_ylabel("Error drop")
        ax.set_title(short_label, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=_GRID_ALPHA)
        _panel_label(ax, panel_tag)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Marginal improvement contributed by each newly selected sample", fontsize=15, y=0.995)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_run_iteration_gradient_corner(
    run_dir: Path,
    output_path: Path,
) -> Path:
    run_dir = Path(run_dir).resolve()
    training_summary = _load_json(run_process_path(run_dir, "training_point_summary.json"))
    initial = np.asarray(training_summary.get("initial_raw_thetas", []), dtype=np.float64)
    if initial.ndim == 1 and initial.size > 0:
        initial = initial.reshape(1, -1)
    selected_batches = list(training_summary.get("selected_raw_thetas_by_iteration", []))
    iterative_rows: list[np.ndarray] = []
    for batch in selected_batches:
        arr = np.asarray(batch, dtype=np.float64)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] > 0:
            iterative_rows.append(arr)

    point_groups = [arr for arr in [initial] + iterative_rows if arr.size > 0]
    if not point_groups:
        raise ValueError("No training points found in training_point_summary.json.")

    all_points = np.vstack(point_groups).astype(np.float64)
    theta_dim = int(all_points.shape[1])
    labels = _resolve_parameter_labels(training_summary, theta_dim)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_points.shape[0] <= theta_dim:
        fig, ax = plt.subplots(figsize=(8.5, 2.8))
        ax.axis("off")
        ax.text(
            0.01,
            0.6,
            (
                "Corner plot downgraded: not enough samples.\n"
                f"samples={all_points.shape[0]}, dims={all_points.shape[1]}"
            ),
            fontsize=11,
            va="center",
            ha="left",
        )
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path

    figure = corner.corner(
        all_points,
        labels=labels,
        quantiles=None,
        show_titles=False,
        color="#777777",
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        range=[0.995] * theta_dim,
        hist_kwargs={"linewidth": 0.9, "color": "#888888"},
    )

    norm = mcolors.Normalize(vmin=1, vmax=max(len(iterative_rows), 1))
    cmap = matplotlib.colormaps["viridis"]
    first_lower_idx: int | None = None
    if len(figure.axes) >= theta_dim * theta_dim:
        for row in range(theta_dim):
            for col in range(theta_dim):
                if col >= row:
                    continue
                ax = figure.axes[row * theta_dim + col]
                if first_lower_idx is None:
                    first_lower_idx = row * theta_dim + col
                if initial.size > 0:
                    ax.scatter(
                        initial[:, col],
                        initial[:, row],
                        c="#B0B0B0",
                        s=24,
                        alpha=0.7,
                        marker="o",
                        edgecolors="none",
                        label="Initial design" if row * theta_dim + col == first_lower_idx else None,
                        zorder=5,
                    )
                for iter_idx, batch in enumerate(iterative_rows, start=1):
                    color = cmap(norm(iter_idx))
                    ax.scatter(
                        batch[:, col],
                        batch[:, row],
                        c=[color],
                        s=30,
                        alpha=0.95,
                        marker="o",
                        edgecolors="white",
                        linewidths=0.25,
                        zorder=6,
                    )
        if first_lower_idx is not None:
            handles, legend_labels = figure.axes[first_lower_idx].get_legend_handles_labels()
            if handles:
                figure.legend(
                    handles,
                    legend_labels,
                    loc="upper right",
                    bbox_to_anchor=(0.975, 0.955),
                    fontsize=9,
                    frameon=True,
                    borderpad=0.35,
                    handletextpad=0.4,
                )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = figure.add_axes([0.29, 0.045, 0.42, 0.02])
    cbar = figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Iteration index")
    figure.suptitle("Sampling trajectory in parameter space", fontsize=15, y=0.995)
    figure.set_size_inches(12.1, 11.5)
    figure.subplots_adjust(top=0.96, bottom=0.11, left=0.07, right=0.985, hspace=0.04, wspace=0.04)

    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def generate_budget_sweep_plots(
    summary_json_path: Path,
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    deep_run_budgets: set[int] | None = None,
) -> dict[str, Any]:
    summary_json_path = Path(summary_json_path).resolve()
    records = _records_from_summary(summary_json_path)
    if not records:
        raise ValueError(f"No records found in {summary_json_path}.")

    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else summary_json_path.parent.resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {
        "budget_metric_triptych": str(
            plot_budget_metric_triptych(records, resolved_output_dir / "budget_metric_triptych.png")
        ),
        "budget_relative_gain_triptych": str(
            plot_budget_gain_triptych(records, resolved_output_dir / "budget_relative_gain_triptych.png")
        ),
        "budget_band_error_bars": str(
            plot_budget_band_error_bars(records, resolved_output_dir / "budget_band_error_bars.png")
        ),
        "budget_p68_gradient_comparison": str(
            plot_budget_p68_gradient(records, resolved_output_dir / "budget_p68_gradient_comparison.png")
        ),
    }

    if deep_run_budgets:
        if config_path is None:
            raise ValueError("config_path is required when generating deep run-level plots.")
        config = load_config(Path(config_path).resolve(), project_root=PROJECT_ROOT)
        per_run_outputs: dict[str, Any] = {}
        for item in records:
            budget = int(item["budget"])
            if budget not in deep_run_budgets:
                continue
            run_dir = Path(item["run_dir"]).resolve()
            run_output_dir = (resolved_output_dir / f"budget_{budget:03d}").resolve()
            run_output_dir.mkdir(parents=True, exist_ok=True)
            trace_json_path = run_output_dir / "iteration_metric_trace.json"
            trace_payload = _rebuild_iteration_metric_trace(
                config,
                active_results_path=Path(item["active_results_path"]).resolve(),
                run_dir=run_dir,
                cache_json_path=trace_json_path,
            )
            per_run_outputs[str(budget)] = {
                "iteration_trace_json": str(trace_json_path),
                "iteration_error_decline": str(
                    plot_run_iteration_error_decline(
                        trace_payload,
                        run_output_dir / "iteration_error_decline.png",
                    )
                ),
                "iteration_marginal_gain": str(
                    plot_run_iteration_marginal_gain(
                        trace_payload,
                        run_output_dir / "iteration_marginal_gain.png",
                    )
                ),
                "corner_iteration_gradient": str(
                    plot_run_iteration_gradient_corner(
                        run_dir,
                        run_output_dir / "corner_iteration_gradient.png",
                    )
                ),
            }
        outputs["per_run_outputs"] = per_run_outputs

    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sweep-specific convergence and run diagnostics plots.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--deep-run-budgets",
        type=str,
        default="",
        help="Comma-separated budgets for heavy per-run reconstruction plots.",
    )
    return parser.parse_args(argv)


def _parse_budget_set(value: str) -> set[int]:
    out: set[int] = set()
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_budget_sweep_plots(
        args.summary_json,
        config_path=args.config,
        output_dir=args.output_dir,
        deep_run_budgets=_parse_budget_set(args.deep_run_budgets),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
