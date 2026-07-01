"""Generate p68-focused plots for the Quijote budget sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    32: "artifacts/reports/runs/20260513_202727",
    64: "artifacts/reports/runs/20260513_204231",
    96: "artifacts/reports/runs/20260513_211404",
    128: "artifacts/reports/runs/20260513_221645",
}
DEFAULT_OUTPUT_DIR = "artifacts/reports/sweeps/quijote_budget_20260513_202620/p68_figures"

MODE_SPECS = {
    "active": {
        "label": "Active learning",
        "subdir": "active_learning_validation",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "fixed": {
        "label": "Fixed-budget GP",
        "subdir": "fixed_budget_comparison",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}

BAND_SPECS = [
    ("low", "0.01-0.1", 0.01, 0.1),
    ("mid", "0.1-1", 0.1, 1.0),
    ("focus_high", "1-5", 1.0, 5.0),
    ("tail", "5-10", 5.0, 10.0),
]

CORE_METRICS = [
    ("overall_p68", "Overall p68"),
    ("focus_integrated_p68", "0.1-5 integrated p68"),
    ("k_le_1_p68k_mean", "Mean p68(k), k<=1"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload.get(key, []), dtype=np.float64)


def _finite_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _band_mask(k: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi >= 10.0:
        return (k >= lo) & (k <= hi)
    return (k >= lo) & (k < hi)


def _payload_path(run_dir: Path, mode: str) -> Path:
    return run_dir / "results" / MODE_SPECS[mode]["subdir"] / "test_set_results.json"


def _summary_path(run_dir: Path, mode: str) -> Path:
    return run_dir / "results" / MODE_SPECS[mode]["subdir"] / "validation_summary.json"


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / "results" / "run_metadata.json"


def _extract_metrics(budget: int, run_dir: Path, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    k = _as_array(payload, "k_bins")
    p68k = _as_array(payload, "power_relative_error_p68")
    p95k = _as_array(payload, "power_relative_error_p95")
    mean_k = _as_array(payload, "power_relative_error_mean")
    sample_mean = _as_array(payload, "sample_mean_relative_error")
    focus_samples = _as_array(payload, "focus_0p1_5_integrated_relative_error_per_sample")

    metrics: dict[str, Any] = {
        "budget": budget,
        "mode": mode,
        "mode_label": MODE_SPECS[mode]["label"],
        "run_dir": str(run_dir),
        "test_set_size": int(payload.get("test_set_size", 0)),
        "theta_dim": int(payload.get("theta_dim", 0)),
        "k_bins": int(k.size),
        "overall_mean": float(payload.get("overall_mean_relative_error", float("nan"))),
        "overall_p68": float(payload.get("overall_p68_relative_error", float("nan"))),
        "overall_p95": float(payload.get("overall_p95_relative_error", float("nan"))),
        "overall_max": float(payload.get("overall_max_relative_error", float("nan"))),
        "sample_mean_p50": _finite_percentile(sample_mean, 50),
        "sample_mean_p68": _finite_percentile(sample_mean, 68),
        "sample_mean_p95": _finite_percentile(sample_mean, 95),
        "focus_integrated_mean": float(
            payload.get("focus_0p1_5_integrated_relative_error_mean", float("nan"))
        ),
        "focus_integrated_p50": float(
            payload.get("focus_0p1_5_integrated_relative_error_p50", float("nan"))
        ),
        "focus_integrated_p68": float(
            payload.get("focus_0p1_5_integrated_relative_error_p68", float("nan"))
        ),
        "focus_integrated_p95": float(
            payload.get("focus_0p1_5_integrated_relative_error_p95", float("nan"))
        ),
        "focus_sample_p68_recomputed": _finite_percentile(focus_samples, 68),
    }

    le_1 = k <= 1.0
    metrics["k_le_1_p68k_mean"] = _finite_mean(p68k[le_1])
    metrics["k_le_1_p68k_median"] = _finite_percentile(p68k[le_1], 50)
    metrics["k_le_1_meank_mean"] = _finite_mean(mean_k[le_1])
    metrics["k_le_1_p95k_mean"] = _finite_mean(p95k[le_1])

    for key, _label, lo, hi in BAND_SPECS:
        mask = _band_mask(k, lo, hi)
        metrics[f"band_{key}_p68k_mean"] = _finite_mean(p68k[mask])
        metrics[f"band_{key}_p68k_median"] = _finite_percentile(p68k[mask], 50)
        metrics[f"band_{key}_meank_mean"] = _finite_mean(mean_k[mask])
        metrics[f"band_{key}_p95k_mean"] = _finite_mean(p95k[mask])
        metrics[f"band_{key}_bin_count"] = int(np.count_nonzero(mask))

    return metrics


def _parse_runs(values: list[str]) -> dict[int, Path]:
    if not values:
        return {budget: PROJECT_ROOT / rel for budget, rel in DEFAULT_RUNS.items()}
    runs: dict[int, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Run spec must be BUDGET=RUN_DIR, got {raw!r}.")
        budget_text, path_text = raw.split("=", 1)
        budget = int(budget_text.strip())
        path = Path(path_text.strip())
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        runs[budget] = path
    return dict(sorted(runs.items()))


def _load_all(run_map: dict[int, Path]) -> tuple[list[dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    flat_records: list[dict[str, Any]] = []
    payloads: dict[int, dict[str, dict[str, Any]]] = {}
    for budget, run_dir in sorted(run_map.items()):
        run_dir = run_dir.resolve()
        payloads[budget] = {}
        metadata = _load_json(_metadata_path(run_dir))
        for mode in MODE_SPECS:
            payload = _load_json(_payload_path(run_dir, mode))
            summary = _load_json(_summary_path(run_dir, mode))
            payloads[budget][mode] = payload
            metrics = _extract_metrics(budget, run_dir, mode, payload)
            metrics["data_source"] = metadata.get("data_source")
            metrics["parameter_space"] = metadata.get("parameter_space")
            metrics["target_transform"] = metadata.get("target_transform")
            metrics["completed_iterations"] = metadata.get("completed_iterations")
            metrics["summary_overall_p68"] = summary.get("overall_p68_relative_error")
            flat_records.append(metrics)
    return flat_records, payloads


def _paired_records(flat_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    for record in flat_records:
        by_budget.setdefault(int(record["budget"]), {})[str(record["mode"])] = record

    rows: list[dict[str, Any]] = []
    metric_keys = [
        "overall_mean",
        "overall_p68",
        "overall_p95",
        "overall_max",
        "focus_integrated_mean",
        "focus_integrated_p68",
        "focus_integrated_p95",
        "k_le_1_p68k_mean",
        "k_le_1_p95k_mean",
        "sample_mean_p68",
    ]
    metric_keys.extend(f"band_{key}_p68k_mean" for key, *_ in BAND_SPECS)
    for budget, modes in sorted(by_budget.items()):
        active = modes["active"]
        fixed = modes["fixed"]
        row: dict[str, Any] = {
            "budget": budget,
            "run_dir": active["run_dir"],
            "data_source": active["data_source"],
            "parameter_space": active["parameter_space"],
            "target_transform": active["target_transform"],
            "theta_dim": active["theta_dim"],
            "k_bins": active["k_bins"],
            "test_set_size": active["test_set_size"],
            "completed_iterations": active["completed_iterations"],
        }
        for key in metric_keys:
            a_val = float(active.get(key, float("nan")))
            f_val = float(fixed.get(key, float("nan")))
            row[f"active_{key}"] = a_val
            row[f"fixed_{key}"] = f_val
            row[f"gain_abs_{key}"] = f_val - a_val
            row[f"gain_pct_{key}"] = 100.0 * (f_val - a_val) / f_val if f_val else float("nan")
        rows.append(row)
    return rows


def _write_tables(output_dir: Path, flat_records: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quijote_p68_budget_summary.json"
    json_path.write_text(
        json.dumps({"records": flat_records, "paired_records": paired}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "quijote_p68_budget_summary.csv"
    fieldnames = sorted({key for row in paired for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in paired:
            writer.writerow(row)
    return {"summary_json": str(json_path), "summary_csv": str(csv_path)}


def _budget_arrays(paired: list[dict[str, Any]], metric_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    budgets = np.asarray([int(row["budget"]) for row in paired], dtype=np.float64)
    active = np.asarray([float(row[f"active_{metric_key}"]) for row in paired], dtype=np.float64)
    fixed = np.asarray([float(row[f"fixed_{metric_key}"]) for row in paired], dtype=np.float64)
    return budgets, active, fixed


def _style(ax: plt.Axes, *, logx: bool = False, logy: bool = False) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.14)
    ax.tick_params(labelsize=10)


def _plot_mode_lines(ax: plt.Axes, budgets: np.ndarray, active: np.ndarray, fixed: np.ndarray) -> None:
    ax.plot(
        budgets,
        100.0 * active,
        color=MODE_SPECS["active"]["color"],
        marker=MODE_SPECS["active"]["marker"],
        linestyle=MODE_SPECS["active"]["linestyle"],
        linewidth=2.4,
        label=MODE_SPECS["active"]["label"],
    )
    ax.plot(
        budgets,
        100.0 * fixed,
        color=MODE_SPECS["fixed"]["color"],
        marker=MODE_SPECS["fixed"]["marker"],
        linestyle=MODE_SPECS["fixed"]["linestyle"],
        linewidth=2.2,
        label=MODE_SPECS["fixed"]["label"],
    )
    ax.set_xticks(budgets)


def plot_p68_dashboard(paired: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    axes_flat = list(axes.flat)
    for ax, (metric_key, label) in zip(axes_flat[:3], CORE_METRICS, strict=True):
        budgets, active, fixed = _budget_arrays(paired, metric_key)
        _plot_mode_lines(ax, budgets, active, fixed)
        _style(ax)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Total training budget")
        ax.set_ylabel("Relative error (%)")

        for x, a_val, f_val in zip(budgets, active, fixed, strict=True):
            gain = 100.0 * (f_val - a_val) / f_val if f_val else np.nan
            ax.annotate(
                f"{gain:+.1f}%",
                xy=(x, 100.0 * min(a_val, f_val)),
                xytext=(0, -18),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#333333",
            )

    ax_gain = axes_flat[3]
    budgets = np.asarray([int(row["budget"]) for row in paired], dtype=np.float64)
    x = np.arange(len(budgets), dtype=np.float64)
    width = 0.24
    colors = ["#4C78A8", "#59A14F", "#B07AA1"]
    for offset, (metric_key, label), color in zip([-width, 0.0, width], CORE_METRICS, colors, strict=True):
        gain = np.asarray([float(row[f"gain_pct_{metric_key}"]) for row in paired], dtype=np.float64)
        ax_gain.bar(x + offset, gain, width=width, color=color, label=label)
    ax_gain.axhline(0.0, color="#222222", linewidth=1.0)
    ax_gain.set_xticks(x, [str(int(item)) for item in budgets])
    ax_gain.set_xlabel("Total training budget")
    ax_gain.set_ylabel("Active improvement vs fixed (%)")
    ax_gain.set_title("p68 improvement summary", fontsize=13)
    _style(ax_gain)
    ax_gain.legend(frameon=False, fontsize=9, loc="best")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.982))
    fig.suptitle("Quijote budget sweep: p68-focused validation errors", fontsize=16, y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_p68_dashboard_compact(paired: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    for ax, (metric_key, label) in zip(axes[:3], CORE_METRICS, strict=True):
        budgets, active, fixed = _budget_arrays(paired, metric_key)
        _plot_mode_lines(ax, budgets, active, fixed)
        _style(ax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Error (%)")

    ax_gain = axes[3]
    budgets = np.asarray([int(row["budget"]) for row in paired], dtype=np.float64)
    x = np.arange(len(budgets), dtype=np.float64)
    width = 0.24
    colors = ["#4C78A8", "#59A14F", "#B07AA1"]
    for offset, (metric_key, label), color in zip([-width, 0.0, width], CORE_METRICS, colors, strict=True):
        gain = np.asarray([float(row[f"gain_pct_{metric_key}"]) for row in paired], dtype=np.float64)
        ax_gain.bar(x + offset, gain, width=width, color=color, label=label)
    ax_gain.axhline(0.0, color="#222222", linewidth=1.0)
    ax_gain.set_xticks(x, [str(int(item)) for item in budgets])
    ax_gain.set_xlabel("Budget")
    ax_gain.set_ylabel("Improvement (%)")
    ax_gain.set_title("Active vs fixed", fontsize=11)
    _style(ax_gain)
    ax_gain.legend(frameon=False, fontsize=7.8, loc="best")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Quijote p68 budget sweep", fontsize=13, y=1.13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overall_context(paired: list[dict[str, Any]], output_path: Path) -> Path:
    metrics = [
        ("overall_mean", "mean"),
        ("overall_p68", "p68"),
        ("overall_p95", "p95"),
        ("overall_max", "max"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    budgets = np.asarray([int(row["budget"]) for row in paired], dtype=np.float64)
    for ax, mode in zip(axes, ["active", "fixed"], strict=True):
        for metric_key, label in metrics:
            values = np.asarray([float(row[f"{mode}_{metric_key}"]) for row in paired], dtype=np.float64)
            alpha = 1.0 if metric_key == "overall_p68" else 0.72
            linewidth = 2.8 if metric_key == "overall_p68" else 1.8
            ax.plot(budgets, 100.0 * values, marker="o", linewidth=linewidth, alpha=alpha, label=label)
        ax.set_title(MODE_SPECS[mode]["label"], fontsize=13)
        ax.set_xlabel("Total training budget")
        ax.set_xticks(budgets)
        _style(ax, logy=True)
    axes[0].set_ylabel("Overall relative error (%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Mean, p68, p95, and max context for overall error", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_p68_spectrum(payloads: dict[int, dict[str, dict[str, Any]]], output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), sharex=True, sharey=True)
    for ax, budget in zip(axes.flat, sorted(payloads), strict=True):
        for mode in ["active", "fixed"]:
            payload = payloads[budget][mode]
            k = _as_array(payload, "k_bins")
            p68k = _as_array(payload, "power_relative_error_p68")
            p95k = _as_array(payload, "power_relative_error_p95")
            color = MODE_SPECS[mode]["color"]
            ax.plot(
                k,
                100.0 * p68k,
                color=color,
                linewidth=2.0,
                linestyle=MODE_SPECS[mode]["linestyle"],
                label=f"{MODE_SPECS[mode]['label']} p68",
            )
            ax.plot(k, 100.0 * p95k, color=color, linewidth=1.0, alpha=0.22)
        ax.axvspan(0.1, 5.0, color="#777777", alpha=0.08, linewidth=0)
        ax.axvline(1.0, color="#444444", linewidth=0.9, linestyle=":", alpha=0.7)
        ax.set_title(f"Budget {budget}", fontsize=13)
        _style(ax, logx=True, logy=True)
    axes[1, 0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[1, 1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0, 0].set_ylabel("Relative error (%)")
    axes[1, 0].set_ylabel("Relative error (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.982))
    fig.suptitle("Per-k p68 relative error spectra; faint lines show p95 context", fontsize=16, y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_focus_sample_distributions(payloads: dict[int, dict[str, dict[str, Any]]], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    colors = plt.cm.viridis(np.linspace(0.12, 0.86, len(payloads)))

    for ax, mode in zip(axes[:2], ["active", "fixed"], strict=True):
        for color, budget in zip(colors, sorted(payloads), strict=True):
            samples = 100.0 * _as_array(
                payloads[budget][mode],
                "focus_0p1_5_integrated_relative_error_per_sample",
            )
            samples = np.sort(samples[np.isfinite(samples)])
            if samples.size == 0:
                continue
            y = np.linspace(1.0 / samples.size, 1.0, samples.size)
            ax.plot(samples, y, color=color, linewidth=2.0, label=f"{budget}")
            ax.scatter(np.percentile(samples, 68), 0.68, color=color, s=28, zorder=5)
        ax.axhline(0.68, color="#222222", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Integrated relative error, 0.1<=k<=5 (%)")
        ax.set_title(MODE_SPECS[mode]["label"], fontsize=13)
        _style(ax)
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(title="Budget", frameon=False, fontsize=8.5, title_fontsize=9)

    positions: list[float] = []
    data: list[np.ndarray] = []
    labels: list[str] = []
    for idx, budget in enumerate(sorted(payloads)):
        base = float(idx + 1)
        for offset, mode in [(-0.16, "active"), (0.16, "fixed")]:
            samples = 100.0 * _as_array(
                payloads[budget][mode],
                "focus_0p1_5_integrated_relative_error_per_sample",
            )
            data.append(samples[np.isfinite(samples)])
            positions.append(base + offset)
        labels.append(str(budget))
    box = axes[2].boxplot(
        data,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
    )
    for idx, patch in enumerate(box["boxes"]):
        mode = "active" if idx % 2 == 0 else "fixed"
        patch.set_facecolor(MODE_SPECS[mode]["color"])
        patch.set_alpha(0.42)
        patch.set_edgecolor(MODE_SPECS[mode]["color"])
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.2)
    axes[2].set_xticks(np.arange(1, len(payloads) + 1), labels)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Total training budget")
    axes[2].set_title("Per-sample focus-window spread", fontsize=13)
    axes[2].set_ylabel("Integrated relative error (%)")
    _style(axes[2])

    fig.suptitle("Distribution of per-sample integrated errors in the 0.1<=k<=5 focus window", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_band_heatmaps(paired: list[dict[str, Any]], output_path: Path) -> Path:
    budgets = [int(row["budget"]) for row in paired]
    band_keys = [key for key, *_ in BAND_SPECS]
    band_labels = [label for _key, label, *_ in BAND_SPECS]

    active = np.asarray(
        [[float(row[f"active_band_{key}_p68k_mean"]) for key in band_keys] for row in paired],
        dtype=np.float64,
    )
    fixed = np.asarray(
        [[float(row[f"fixed_band_{key}_p68k_mean"]) for key in band_keys] for row in paired],
        dtype=np.float64,
    )
    gain = 100.0 * (fixed - active) / fixed

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0))
    panels = [
        (100.0 * active, "Active p68(k) mean (%)", "magma_r", None, None),
        (100.0 * fixed, "Fixed p68(k) mean (%)", "magma_r", None, None),
        (gain, "Active improvement (%)", "RdYlGn", -35.0, 35.0),
    ]
    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(band_labels)), band_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(budgets)), [str(item) for item in budgets])
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"$k$ band")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text = f"{value:.2f}" if title != "Active improvement (%)" else f"{value:+.1f}"
                ax.text(col, row, text, ha="center", va="center", fontsize=8.8, color="#111111")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Total training budget")
    fig.suptitle("Band-averaged p68(k) relative error from per-k validation curves", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_gain_waterfall(paired: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), sharex=True)
    panels = [
        ("overall_p68", "Overall p68"),
        ("focus_integrated_p68", "0.1-5 integrated p68"),
        ("sample_mean_p68", "Sample-mean p68"),
        ("overall_p95", "Overall p95 context"),
    ]
    budgets = np.asarray([int(row["budget"]) for row in paired], dtype=np.float64)
    for ax, (metric_key, title) in zip(axes.flat, panels, strict=True):
        active = np.asarray([float(row[f"active_{metric_key}"]) for row in paired], dtype=np.float64)
        fixed = np.asarray([float(row[f"fixed_{metric_key}"]) for row in paired], dtype=np.float64)
        gain = fixed - active
        colors = ["#009E73" if item >= 0 else "#C7362F" for item in gain]
        ax.bar(budgets, 100.0 * gain, width=12.0, color=colors, alpha=0.88)
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Fixed - active error (percentage points)")
        ax.set_xticks(budgets)
        _style(ax)
        for x, y in zip(budgets, gain, strict=True):
            ax.annotate(
                f"{100.0 * y:+.2f}",
                xy=(x, 100.0 * y),
                xytext=(0, 5 if y >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
    axes[1, 0].set_xlabel("Total training budget")
    axes[1, 1].set_xlabel("Total training budget")
    fig.suptitle("Absolute p68-focused gains: positive means active learning is better", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_plots(run_map: dict[int, Path], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_records, payloads = _load_all(run_map)
    paired = _paired_records(flat_records)
    outputs: dict[str, Any] = _write_tables(output_dir, flat_records, paired)
    outputs.update(
        {
            "p68_dashboard": str(plot_p68_dashboard(paired, output_dir / "quijote_p68_budget_dashboard.png")),
            "p68_dashboard_compact": str(
                plot_p68_dashboard_compact(paired, output_dir / "quijote_p68_budget_dashboard_compact.png")
            ),
            "overall_context": str(plot_overall_context(paired, output_dir / "quijote_error_stat_context.png")),
            "p68_spectrum": str(plot_p68_spectrum(payloads, output_dir / "quijote_p68_spectrum_by_budget.png")),
            "focus_sample_distributions": str(
                plot_focus_sample_distributions(
                    payloads,
                    output_dir / "quijote_focus_sample_error_distributions.png",
                )
            ),
            "band_heatmaps": str(plot_band_heatmaps(paired, output_dir / "quijote_p68_band_heatmaps.png")),
            "gain_waterfall": str(plot_gain_waterfall(paired, output_dir / "quijote_p68_gain_waterfall.png")),
        }
    )
    manifest_path = output_dir / "quijote_p68_plot_manifest.json"
    manifest_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Budget/run mapping as BUDGET=RUN_DIR. Defaults to the 20260513 Quijote sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots and tables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_map = _parse_runs(args.run)
    outputs = generate_plots(run_map, args.output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
