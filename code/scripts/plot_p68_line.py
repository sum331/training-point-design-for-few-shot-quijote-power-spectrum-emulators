from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from z2quijote.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the z2 primary p68(k) line only.")
    parser.add_argument("--predictions", required=True, help="Prediction npz with k_bins and truth/pred arrays.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.yaml"))
    parser.add_argument("--summary", default=None, help="Optional run summary JSON used for title metadata.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults next to the predictions file.")
    parser.add_argument("--curve-npz", default=None, help="Optional output npz for k and p68(k).")
    parser.add_argument("--curve-json", default=None, help="Optional p68-only summary JSON path.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predictions_path = Path(args.predictions).resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"prediction file not found: {predictions_path}")

    with np.load(predictions_path) as data:
        k_bins = np.asarray(data["k_bins"], dtype=np.float64)
        abs_relative_error = None
        if "signed_relative_bias" in data.files:
            abs_relative_error = np.abs(np.asarray(data["signed_relative_bias"], dtype=np.float64))
        elif "truth_target" in data.files and "pred_target" in data.files:
            truth = np.asarray(data["truth_target"], dtype=np.float64)
            pred = np.asarray(data["pred_target"], dtype=np.float64)
            abs_relative_error = np.abs(np.exp(pred - truth) - 1.0)

        if "kwise_p68_relative_error" in data.files:
            p68_line = np.asarray(data["kwise_p68_relative_error"], dtype=np.float64)
        elif abs_relative_error is not None:
            p68_line = np.percentile(abs_relative_error, 68.0, axis=0)
        else:
            raise KeyError("prediction file must contain kwise_p68_relative_error, signed_relative_bias, or truth/pred arrays")

    if k_bins.shape != p68_line.shape:
        raise ValueError(f"k_bins and p68 line shapes differ: {k_bins.shape} vs {p68_line.shape}")

    output_path = Path(args.output).resolve() if args.output else predictions_path.with_name("sobol64_lhs256_p68_line.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve_npz_path = Path(args.curve_npz).resolve() if args.curve_npz else output_path.with_suffix(".npz")
    curve_json_path = Path(args.curve_json).resolve() if args.curve_json else output_path.with_suffix(".json")
    np.savez_compressed(
        curve_npz_path,
        k_bins=k_bins.astype(np.float64),
        kwise_p68_relative_error=p68_line.astype(np.float64),
    )

    summary = _load_summary(args.summary)
    _plot_p68_line(
        k_bins=k_bins,
        p68_line=p68_line,
        output_path=output_path,
        band_edges=config.evaluation.band_edges,
        title_suffix=_title_suffix(summary),
    )
    result = {
        "output_path": str(output_path),
        "curve_npz_path": str(curve_npz_path),
        "primary_metric": config.evaluation.primary_metric,
        "primary_curve": config.evaluation.primary_curve,
        "report_metric_policy": config.evaluation.report_metric_policy,
        "overall_p68_relative_error": (
            float(np.percentile(abs_relative_error, 68.0)) if abs_relative_error is not None else None
        ),
        "band_p68_relative_error": (
            _band_p68(k_bins, abs_relative_error, config.evaluation.band_edges, config.evaluation.band_labels)
            if abs_relative_error is not None
            else {}
        ),
        "overall_line_p68_mean": float(np.mean(p68_line)),
        "line_min": float(np.min(p68_line)),
        "line_max": float(np.max(p68_line)),
        "auxiliary_metrics_excluded_from_report": ["p50", "p95", "mean", "max", "signed_bias"],
    }
    curve_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["curve_json_path"] = str(curve_json_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _plot_p68_line(
    *,
    k_bins: np.ndarray,
    p68_line: np.ndarray,
    output_path: Path,
    band_edges: tuple[float, ...],
    title_suffix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(k_bins, 100.0 * p68_line, color="#1864ab", linewidth=2.2, label="p68(k)")
    for edge in band_edges:
        ax.axvline(float(edge), color="#868e96", linewidth=0.8, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$p68(|\exp(\Delta\log P)-1|)$ [%]")
    ax.set_title(f"Sobol64 / LHS256 Quijote fixed GP: p68(k) only{title_suffix}")
    ax.grid(True, which="major", color="#dee2e6", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#f1f3f5", linewidth=0.5)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _load_summary(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text).resolve()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _band_p68(
    k_bins: np.ndarray,
    abs_relative_error: np.ndarray,
    band_edges: tuple[float, ...],
    band_labels: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    lower = -np.inf
    labels = band_labels or tuple(f"band_{index}" for index in range(len(band_edges) + 1))
    for label, upper in zip(labels, tuple(band_edges) + (np.inf,), strict=True):
        mask = (k_bins >= lower) & (k_bins < float(upper))
        if np.any(mask):
            result[str(label)] = float(np.percentile(abs_relative_error[:, mask], 68.0))
        lower = float(upper)
    return result


def _title_suffix(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    train = summary.get("train", {})
    validation = summary.get("validation", {})
    train_seed = train.get("seed")
    validation_seed = validation.get("seed")
    if train_seed is None or validation_seed is None:
        return ""
    return f"  seeds {train_seed}/{validation_seed}"


if __name__ == "__main__":
    raise SystemExit(main())
