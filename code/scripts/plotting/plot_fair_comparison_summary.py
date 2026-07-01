"""Plot a compact conclusion suite for z2 fair-comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_DEFAULT_OUTPUT_DIRNAME = "figures"
_DESIGN_ORDER = (
    "ppr32_plus_z2_active32",
    "sobol64",
)
_DESIGN_LABELS = {
    "ppr32": "PPR32",
    "ppr32_plus_z2_active32": "PPR32 + Z2 AL32",
    "sobol64": "Sobol64",
    "ppr32_plus_sobol32": "PPR32 + Sobol32",
}
_DESIGN_COLORS = {
    "ppr32": "#7B8CDE",
    "ppr32_plus_z2_active32": "#2A9D8F",
    "sobol64": "#E76F51",
    "ppr32_plus_sobol32": "#264653",
}
_DESIGN_MARKERS = {
    "ppr32": "o",
    "ppr32_plus_z2_active32": "s",
    "sobol64": "^",
    "ppr32_plus_sobol32": "D",
}


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric_value(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                pass
    return float(default)


def _resolve_design_label(name: str, training_points: int | None) -> str:
    base = _DESIGN_LABELS.get(name, name)
    if training_points is None:
        return base
    return f"{base} ({training_points} pts)"


def _load_design_points(design_path: Path) -> np.ndarray | None:
    if not design_path.exists():
        return None
    if design_path.suffix.lower() != ".npz":
        return None
    with np.load(design_path, allow_pickle=False) as data:
        if "theta_raw" in data.files:
            return np.asarray(data["theta_raw"], dtype=np.float64)
        if "selected_theta_raw" in data.files:
            return np.asarray(data["selected_theta_raw"], dtype=np.float64)
    return None


def _plot_overall_p68(design_results: dict[str, Any], output_path: Path) -> Path:
    ordered = [name for name in _DESIGN_ORDER if name in design_results]
    if not ordered:
        ordered = sorted(design_results)
    values = []
    labels = []
    for name in ordered:
        metrics = dict(design_results[name].get("metrics", {}))
        overall = metrics.get("overall_relative_error", {})
        values.append(_metric_value(overall, "p68"))
        labels.append(_resolve_design_label(name, int(design_results[name].get("training_points", 0)) or None))

    x = np.arange(len(values), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bars = ax.bar(
        x,
        values,
        color=[_DESIGN_COLORS.get(name, "#4C78A8") for name in ordered],
        alpha=0.92,
    )
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Overall p68 relative error")
    ax.set_title("z2 fair comparison: overall p68")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_band_p68(design_results: dict[str, Any], output_path: Path) -> Path:
    bands = ("low", "mid", "focus_high", "tail")
    ordered = [name for name in _DESIGN_ORDER if name in design_results]
    if not ordered:
        ordered = sorted(design_results)
    x = np.arange(len(bands), dtype=np.float64)
    width = 0.18 if len(ordered) >= 4 else 0.24

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    for index, name in enumerate(ordered):
        metrics = dict(design_results[name].get("metrics", {}))
        bands_payload = dict(metrics.get("bands", {}))
        values = [
            _metric_value(dict(bands_payload.get(band, {})).get("relative_error", {}), "p68")
            for band in bands
        ]
        offset = (index - (len(ordered) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=_DESIGN_COLORS.get(name, "#4C78A8"),
            alpha=0.88,
            label=_resolve_design_label(name, int(design_results[name].get("training_points", 0)) or None),
        )

    ax.set_xticks(x, ["low", "mid", "focus_high", "tail"])
    ax.set_ylabel("Band p68 relative error")
    ax.set_title("z2 fair comparison: band p68")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_improvement_fractions(comparison: dict[str, Any], output_path: Path) -> Path:
    ordered = list(comparison.keys())
    labels = [key.replace("_vs_", "\nvs\n") for key in ordered]
    values = [100.0 * _metric_value(dict(comparison[key]), "p68_improvement_fraction") for key in ordered]
    x = np.arange(len(values), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    colors = ["#2A9D8F" if value >= 0.0 else "#D1495B" for value in values]
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x, labels, rotation=12, ha="center")
    ax.set_ylabel("p68 improvement fraction (%)")
    ax.set_title("z2 fair comparison: improvement vs baselines")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_design_corner(design_results: dict[str, Any], output_path: Path) -> Path:
    if corner is None:
        fig, ax = plt.subplots(figsize=(8.8, 3.0))
        ax.axis("off")
        ax.text(0.01, 0.6, "corner package is not available", fontsize=12, va="center", ha="left")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    point_sets: list[tuple[str, np.ndarray]] = []
    for name in _DESIGN_ORDER:
        entry = design_results.get(name)
        if not entry:
            continue
        design_path = Path(entry.get("design_path", "")).resolve()
        points = _load_design_points(design_path)
        if points is not None and points.size > 0:
            point_sets.append((name, points))

    if not point_sets:
        fig, ax = plt.subplots(figsize=(8.8, 3.0))
        ax.axis("off")
        ax.text(0.01, 0.6, "No design points available for corner plot.", fontsize=12, va="center", ha="left")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    all_points = np.vstack([points for _, points in point_sets]).astype(np.float64)
    ndim = all_points.shape[1]
    figure = corner.corner(
        all_points,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color="#444444",
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        range=[0.995] * ndim,
    )

    first_lower_idx: int | None = None
    if len(figure.axes) >= ndim * ndim:
        for row in range(ndim):
            for col in range(ndim):
                if col >= row:
                    continue
                idx = row * ndim + col
                if first_lower_idx is None:
                    first_lower_idx = idx
                ax = figure.axes[idx]
                for name, points in point_sets:
                    ax.scatter(
                        points[:, col],
                        points[:, row],
                        s=46,
                        alpha=0.9,
                        marker=_DESIGN_MARKERS.get(name, "o"),
                        color=_DESIGN_COLORS.get(name, "#4C78A8"),
                        edgecolors="white",
                        linewidths=0.5,
                        label=_DESIGN_LABELS.get(name, name) if idx == first_lower_idx else None,
                    )
        if first_lower_idx is not None:
            figure.axes[first_lower_idx].legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92)

    figure.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_conclusion(design_results: dict[str, Any], comparison: dict[str, Any], output_path: Path) -> Path:
    ordered = [name for name in _DESIGN_ORDER if name in design_results]
    if not ordered:
        ordered = sorted(design_results)
    if not ordered:
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        ax.axis("off")
        ax.text(0.02, 0.6, "No design results available for conclusion plot.", fontsize=12, va="center", ha="left")
        fig.savefig(output_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return output_path

    primary_name = next((name for name in ordered if "_plus_z2_active" in name), ordered[0])
    seed_name = next((name for name in ordered if name.startswith("ppr") and "_plus_" not in name), ordered[0])
    sobol_name = next((name for name in ordered if name.startswith("sobol") and "training_points" in design_results[name]), None)
    if sobol_name is None:
        sobol_name = next((name for name in ordered if name.startswith("sobol")), ordered[-1])

    primary_entry = dict(design_results.get(primary_name, {}))
    sobol_entry = dict(design_results.get(sobol_name, {}))
    seed_entry = dict(design_results.get(seed_name, {}))
    primary_metrics = dict(primary_entry.get("metrics", {}))
    sobol_metrics = dict(sobol_entry.get("metrics", {}))
    seed_metrics = dict(seed_entry.get("metrics", {}))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18.0, 7.0))

    left_values = [
        100.0 * _metric_value(dict(design_results[name].get("metrics", {})).get("overall_relative_error", {}), "p68")
        for name in ordered
    ]
    left_labels = [_resolve_design_label(name, int(design_results[name].get("training_points", 0)) or None) for name in ordered]
    left_x = np.arange(len(left_values), dtype=np.float64)
    bars = ax_left.bar(
        left_x,
        left_values,
        color=[_DESIGN_COLORS.get(name, "#4C78A8") for name in ordered],
        alpha=0.94,
    )
    ax_left.set_xticks(left_x, left_labels, rotation=15, ha="right")
    ax_left.set_ylabel("Overall p68 [%]")
    ax_left.set_title("Overall comparison")
    ax_left.grid(True, axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, left_values, strict=True):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    sobol_overall = 100.0 * _metric_value(sobol_metrics.get("overall_relative_error", {}), "p68")
    ax_left.axhline(sobol_overall, color=_DESIGN_COLORS.get(sobol_name, "#E76F51"), linestyle="--", linewidth=1.2, alpha=0.8)
    box_lines = []
    for baseline_name in (sobol_name, seed_name):
        key = f"{primary_name}_vs_{baseline_name}"
        block = comparison.get(key)
        if block is not None:
            box_lines.append(
                f"{_resolve_design_label(primary_name, int(primary_entry.get('training_points', 0)) or None)} vs "
                f"{_resolve_design_label(baseline_name, int(design_results[baseline_name].get('training_points', 0)) or None)}: "
                f"{100.0 * float(block.get('p68_improvement_fraction', 0.0)):+.2f}%"
            )
    if box_lines:
        ax_left.text(
            0.02,
            0.98,
            "\n".join(box_lines),
            transform=ax_left.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="#B8C4CF"),
        )

    band_labels = ["Overall", "Low", "Mid", "Focus high", "Tail"]
    band_keys = ("overall_relative_error", "low", "mid", "focus_high", "tail")
    primary_band = []
    sobol_band = []
    for band_key in band_keys:
        if band_key == "overall_relative_error":
            primary_band.append(100.0 * _metric_value(primary_metrics.get(band_key, {}), "p68"))
            sobol_band.append(100.0 * _metric_value(sobol_metrics.get(band_key, {}), "p68"))
            continue
        primary_band.append(
            100.0
            * _metric_value(
                dict(primary_metrics.get("bands", {}).get(band_key, {})).get("relative_error", {}),
                "p68",
            )
        )
        sobol_band.append(
            100.0
            * _metric_value(
                dict(sobol_metrics.get("bands", {}).get(band_key, {})).get("relative_error", {}),
                "p68",
            )
        )

    x = np.arange(len(band_labels), dtype=np.float64)
    width = 0.34
    ax_right.bar(
        x - width / 2.0,
        primary_band,
        width=width,
        color=_DESIGN_COLORS.get(primary_name, "#2A9D8F"),
        alpha=0.92,
        label=_resolve_design_label(primary_name, int(primary_entry.get("training_points", 0)) or None),
    )
    ax_right.bar(
        x + width / 2.0,
        sobol_band,
        width=width,
        color=_DESIGN_COLORS.get(sobol_name, "#E76F51"),
        alpha=0.86,
        label=_resolve_design_label(sobol_name, int(sobol_entry.get("training_points", 0)) or None),
    )
    for idx, value in enumerate(primary_band):
        ax_right.text(
            float(x[idx] - width / 2.0),
            value,
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for idx, value in enumerate(sobol_band):
        ax_right.text(
            float(x[idx] + width / 2.0),
            value,
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_right.set_xticks(x, band_labels)
    ax_right.set_ylabel("P68 [%]")
    ax_right.set_title("Segmented p68 comparison")
    ax_right.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_right.legend(frameon=False, fontsize=10, loc="upper right")
    ax_right.text(
        0.02,
        0.98,
        "Positive % means active learning is better than Sobol64",
        transform=ax_right.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#B8C4CF"),
    )

    fig.suptitle("Z2 Quijote fair comparison, 64 points, p68 only", fontsize=17, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_table(design_results: dict[str, Any], comparison: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "z2_cdm_logdiff_summary_table.csv"
    json_path = output_dir / "z2_cdm_logdiff_summary_table.json"

    rows: list[dict[str, Any]] = []
    for name in _DESIGN_ORDER:
        entry = design_results.get(name)
        if not entry:
            continue
        metrics = dict(entry.get("metrics", {}))
        overall = dict(metrics.get("overall_relative_error", {}))
        row = {
            "design": name,
            "label": _DESIGN_LABELS.get(name, name),
            "training_points": entry.get("training_points"),
            "overall_p68": _metric_value(overall, "p68"),
            "overall_mean": _metric_value(overall, "mean"),
            "overall_p95": _metric_value(overall, "p95"),
            "overall_max": _metric_value(overall, "max"),
        }
        bands = dict(metrics.get("bands", {}))
        for band in ("low", "mid", "focus_high", "tail"):
            band_entry = dict(bands.get(band, {}))
            rel = dict(band_entry.get("relative_error", {}))
            row[f"{band}_p68"] = _metric_value(rel, "p68")
            row[f"{band}_mean"] = _metric_value(rel, "mean")
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return [csv_path, json_path]


def generate_summary_suite(summary_path: Path, output_dir: Path | None = None) -> list[Path]:
    summary_path = Path(summary_path).resolve()
    summary = _load_json(summary_path)
    design_results = dict(summary.get("design_results", {}))
    comparison = dict(summary.get("comparison", {}))

    resolved_output_dir = Path(output_dir).resolve() if output_dir is not None else summary_path.parent / _DEFAULT_OUTPUT_DIRNAME
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    return _write_table(design_results, comparison, resolved_output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write z2 fair-comparison summary tables.")
    parser.add_argument("--summary", type=Path, required=True, help="Run summary JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots and tables.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_summary_suite(args.summary, args.output_dir)
    for path in outputs:
        print(f"[Plot] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
