"""Build manuscript-ready mechanism-ablation table assets for z2quijote.

The table separates cold-start, active-learning, bias, and initial-condition
controls under the same fixed validation protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_REGISTRY = Path(
    r"data\ablation\mechanism_20260701_paper_ablation\ablation_registry.csv"
)
DEFAULT_GALLERY = Path(
    r"data\paper_figures\main_results_20260701\numbered_gallery"
)


@dataclass(frozen=True)
class DesignInfo:
    condition: str
    label: str
    group: str
    acquisition_label: str
    contrast_label: str
    inference: str


DESIGNS: tuple[DesignInfo, ...] = (
    DesignInfo(
        "sobol32",
        "Sobol32",
        "Cold start",
        "fixed",
        "Sobol32 baseline",
        "space-filling reference for 32-point designs",
    ),
    DesignInfo(
        "ppr32",
        "PPR32",
        "Cold start",
        "PPR only",
        "PPR cold start vs Sobol32",
        "bias-prior relaxation improves the initial design",
    ),
    DesignInfo(
        "sobol64",
        "Sobol64",
        "64-point baseline",
        "fixed",
        "Sobol64 baseline",
        "main 64-point space-filling reference",
    ),
    DesignInfo(
        "ppr32_plus_sobol32",
        "PPR32 + Sobol32",
        "64-point control",
        "neutral fill",
        "PPR plus neutral Sobol fill",
        "cold start alone does not explain the final gain",
    ),
    DesignInfo(
        "ppr32_plus_variance_only_al32",
        "PPR32 + variance-only AL32",
        "AL component",
        "variance only",
        "variance-only AL after PPR",
        "posterior variance alone is not sufficient",
    ),
    DesignInfo(
        "ppr32_plus_bias_only_al32",
        "PPR32 + bias-only AL32",
        "AL component",
        "bias only",
        "bias-only AL after PPR",
        "KUN bias proxy adds useful difficult-region signal",
    ),
    DesignInfo(
        "ppr32_plus_variance_bias_al32",
        "PPR32 + variance-bias AL32",
        "Full method",
        "variance + bias",
        "variance-bias AL after PPR",
        "lowest error; variance and bias are complementary",
    ),
    DesignInfo(
        "sobol32_plus_variance_bias_al32",
        "Sobol32 + variance-bias AL32",
        "Initial-condition control",
        "variance + bias",
        "variance-bias AL after Sobol32",
        "same acquisition without PPR does not recover the gain",
    ),
)


PALETTE = {
    "Cold start": "#EEF4FA",
    "64-point baseline": "#F0F0F0",
    "64-point control": "#F5EFE2",
    "AL component": "#EAF5F1",
    "Full method": "#C5E6D3",
    "Initial-condition control": "#F6E8EE",
}


def _read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["condition"]): row for row in rows}


def _as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _improvement(reference: float, value: float) -> float:
    return (reference - value) / reference * 100.0


def _fmt_num(value: float) -> str:
    return f"{value:.5f}"


def _fmt_delta(value: float | None, reference_label: str) -> str:
    if value is None or math.isnan(value):
        return "reference"
    return f"{value:+.2f}% vs {reference_label}"


def _build_rows(registry: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    sobol32 = _as_float(registry["sobol32"]["overall_p68"])
    sobol64 = _as_float(registry["sobol64"]["overall_p68"])
    full = _as_float(registry["ppr32_plus_variance_bias_al32"]["overall_p68"])

    rows: list[dict[str, str]] = []
    for design in DESIGNS:
        raw = registry[design.condition]
        p68 = _as_float(raw["overall_p68"])
        mean = _as_float(raw["overall_mean"])
        p95 = _as_float(raw["overall_p95"])
        if design.condition == "sobol32":
            primary_delta = None
            primary_reference = "Sobol32"
        elif design.condition == "ppr32":
            primary_delta = _improvement(sobol32, p68)
            primary_reference = "Sobol32"
        elif design.condition == "sobol64":
            primary_delta = None
            primary_reference = "Sobol64"
        else:
            primary_delta = _improvement(sobol64, p68)
            primary_reference = "Sobol64"

        full_vs_this = "" if design.condition == "ppr32_plus_variance_bias_al32" else f"{_improvement(p68, full):+.2f}%"
        rows.append(
            {
                "group": design.group,
                "condition": design.condition,
                "design": design.label,
                "n_train": str(raw["training_points"]),
                "initial_design": raw["initial_design"],
                "acquisition": design.acquisition_label,
                "overall_p68": _fmt_num(p68),
                "overall_mean": _fmt_num(mean),
                "overall_p95": _fmt_num(p95),
                "delta_primary": _fmt_delta(primary_delta, primary_reference),
                "delta_full_vs_this": full_vs_this,
                "contrast": design.contrast_label,
                "inference": design.inference,
            }
        )
    return rows


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _write_source_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\footnotesize",
        r"\caption{Mechanism ablation under the fixed Quijote LHS256 validation protocol. Positive \(\Delta p68\) means a lower overall \(p68\) relative error than the stated reference.}",
        r"\label{tab:mechanism_ablation}",
        r"\begin{tabularx}{\textwidth}{p{0.14\textwidth}p{0.20\textwidth}cccp{0.19\textwidth}Xp{0.22\textwidth}}",
        r"\toprule",
        r"Component & Design & \(N_{\rm train}\) & Acquisition & \(p68\) & \(\Delta p68\) & Controlled contrast & Interpretation \\",
        r"\midrule",
    ]
    last_group: str | None = None
    for row in rows:
        if last_group is not None and row["group"] != last_group:
            lines.append(r"\addlinespace[2pt]")
        last_group = row["group"]
        design = r"\textbf{" + _latex_escape(row["design"]) + r"}" if row["condition"] == "ppr32_plus_variance_bias_al32" else _latex_escape(row["design"])
        p68 = r"\textbf{" + row["overall_p68"] + r"}" if row["condition"] == "ppr32_plus_variance_bias_al32" else row["overall_p68"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["group"]),
                    design,
                    row["n_train"],
                    _latex_escape(row["acquisition"]),
                    p68,
                    _latex_escape(row["delta_primary"]),
                    _latex_escape(row["contrast"]),
                    _latex_escape(row["inference"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _draw_linked_table(rows: list[dict[str, str]], out_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
        }
    )

    labels = [row["design"] for row in rows]
    p68 = [float(row["overall_p68"]) for row in rows]
    sobol32 = p68[0]
    sobol64 = p68[2]

    fig = plt.figure(figsize=(8.7, 6.35), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[0.88, 1.72],
        left=0.075,
        right=0.99,
        bottom=0.105,
        top=0.88,
        wspace=0.11,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1])

    colors = [PALETTE[row["group"]] for row in rows]
    y_pos = list(range(len(rows)))[::-1]
    ax_bar.barh(y_pos, p68, color=colors, edgecolor="#565656", linewidth=0.45)
    ax_bar.axvline(sobol32, color="#9E9E9E", linestyle=":", linewidth=1.0, label="Sobol32")
    ax_bar.axvline(sobol64, color="#D55E00", linestyle="--", linewidth=1.0, label="Sobol64")
    ax_bar.set_yticks(y_pos, labels)
    ax_bar.set_xlabel(r"Overall $p68$ relative error")
    ax_bar.set_title("a  Quantitative ablation", loc="left", fontweight="bold", fontsize=9)
    ax_bar.grid(axis="x", color="#E2E2E2", linewidth=0.55)
    ax_bar.set_xlim(0.0, max(p68) * 1.13)
    for y, value, row in zip(y_pos, p68, rows, strict=True):
        weight = "bold" if row["condition"] == "ppr32_plus_variance_bias_al32" else "normal"
        ax_bar.text(value + 0.00035, y, f"{value:.5f}", va="center", ha="left", fontsize=7.0, fontweight=weight)
    ax_bar.legend(loc="lower right", fontsize=6.4, handlelength=1.8, borderaxespad=0.2)

    ax_tab.axis("off")
    ax_tab.set_title("b  Component contribution", loc="left", fontweight="bold", fontsize=9, pad=7)

    headers = ["Controlled contrast", r"$\Delta p68$", "Interpretation"]
    col_x = [0.02, 0.40, 0.58]
    row_h = 0.094
    y_top = 0.91

    ax_tab.add_patch(Rectangle((0.0, y_top), 0.99, 0.06, transform=ax_tab.transAxes, facecolor="#F3F5F7", edgecolor="#C7CDD3", linewidth=0.6))
    for x, text in zip(col_x, headers, strict=True):
        ax_tab.text(x, y_top + 0.031, text, transform=ax_tab.transAxes, va="center", ha="left", fontweight="bold", fontsize=7.3)

    for idx, row in enumerate(rows):
        y = y_top - (idx + 1) * row_h
        face = PALETTE[row["group"]]
        ax_tab.add_patch(
            Rectangle((0.0, y), 0.99, row_h * 0.92, transform=ax_tab.transAxes, facecolor=face, edgecolor="white", linewidth=0.7)
        )
        design = row["design"]
        contrast = row["contrast"]
        answer = row["inference"]
        if row["condition"] == "ppr32_plus_variance_bias_al32":
            design = "Full method"
            answer = "lowest p68; variance and bias act as complementary signals"
        if row["condition"] == "sobol32_plus_variance_bias_al32":
            answer = "initial-condition control; Sobol-seeded AL does not recover the gain"
        contrast = textwrap.fill(contrast, width=34)
        answer = textwrap.fill(answer, width=45)
        text_weight = "bold" if row["condition"] == "ppr32_plus_variance_bias_al32" else "normal"
        ax_tab.text(col_x[0], y + row_h * 0.58, contrast, transform=ax_tab.transAxes, va="center", ha="left", fontsize=6.3, fontweight=text_weight)
        ax_tab.text(col_x[1], y + row_h * 0.58, row["delta_primary"], transform=ax_tab.transAxes, va="center", ha="left", fontsize=6.3)
        ax_tab.text(col_x[2], y + row_h * 0.58, answer, transform=ax_tab.transAxes, va="center", ha="left", fontsize=6.3, fontweight=text_weight)

    fig.suptitle(
        "Component ablation under fixed Quijote LHS256 validation",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.027,
        r"All entries use the same parameter box, residual-anchor target, $N_k=475$ grid and LHS256 validation set. Positive $\Delta p68$ means lower error than the stated reference.",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#555555",
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _draw_single_column_ablation(rows: list[dict[str, str]], out_stem: Path) -> None:
    """Draw a compact single-column ablation panel for Word/two-column layout."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
        }
    )

    short_labels = {
        "sobol32": "Sobol32",
        "ppr32": "PPR32",
        "sobol64": "Sobol64",
        "ppr32_plus_sobol32": "PPR + Sobol",
        "ppr32_plus_variance_only_al32": "PPR + variance",
        "ppr32_plus_bias_only_al32": "PPR + bias",
        "ppr32_plus_variance_bias_al32": "Full method",
        "sobol32_plus_variance_bias_al32": "Sobol + v+b",
    }
    point_colors = {
        "sobol32": "#B8B8B8",
        "ppr32": "#5DA5DA",
        "sobol64": "#E69F00",
        "ppr32_plus_sobol32": "#D7B56D",
        "ppr32_plus_variance_only_al32": "#7AA6A1",
        "ppr32_plus_bias_only_al32": "#4C9F70",
        "ppr32_plus_variance_bias_al32": "#0072B2",
        "sobol32_plus_variance_bias_al32": "#CC79A7",
    }
    labels = [short_labels[row["condition"]] for row in rows]
    values = [float(row["overall_p68"]) for row in rows]
    conditions = [row["condition"] for row in rows]
    sobol32 = values[0]
    sobol64 = values[2]

    fig, ax = plt.subplots(figsize=(3.35, 3.35), constrained_layout=False)
    y_pos = list(range(len(rows)))[::-1]
    x_min = min(values) - 0.0009
    x_max = max(values) + 0.00075

    for y, value, condition in zip(y_pos, values, conditions, strict=True):
        color = point_colors[condition]
        ax.hlines(y, x_min, value, color=color, linewidth=1.1, alpha=0.45)
        marker_size = 46 if condition == "ppr32_plus_variance_bias_al32" else 32
        edge = "black" if condition == "ppr32_plus_variance_bias_al32" else "white"
        ax.scatter(value, y, s=marker_size, color=color, edgecolor=edge, linewidth=0.55, zorder=3)
        ax.text(value + 0.00012, y, f"{value:.5f}", va="center", ha="left", fontsize=6.6)

    ax.axvline(sobol32, color="#8A8A8A", linestyle=":", linewidth=0.9)
    ax.axvline(sobol64, color="#D55E00", linestyle="--", linewidth=0.9)
    ax.text(sobol64 + 0.00003, len(rows) - 0.12, "Sobol64", rotation=90, va="top", ha="left", fontsize=6.1, color="#A85000")

    ax.set_yticks(y_pos, labels)
    ax.set_xlabel(r"Overall $p68$ relative error")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Component ablation", loc="left", fontsize=8.5, fontweight="bold")
    ax.text(
        0.0,
        -0.26,
        r"Lower is better; all entries use LHS256 and $N_k=475$.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.25,
        color="#555555",
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_assets(registry_path: Path, gallery_root: Path) -> list[Path]:
    registry = _read_registry(registry_path)
    rows = _build_rows(registry)

    source_csv = gallery_root / "source_data" / "Table04_mechanism_ablation_source.csv"
    table_tex = gallery_root / "tables" / "Table04_mechanism_ablation_table04_mechanism_ablation.tex"
    table_csv = gallery_root / "tables" / "Table04_mechanism_ablation_table04_mechanism_ablation_source.csv"
    figure_stem = gallery_root / "figures" / "Fig14_mechanism_ablation_linked_table"
    single_column_stem = gallery_root / "figures" / "Fig08_component_ablation_single_column"

    _write_source_csv(rows, source_csv)
    _write_source_csv(rows, table_csv)
    _write_latex(rows, table_tex)
    _draw_linked_table(rows, figure_stem)
    _draw_single_column_ablation(rows, single_column_stem)

    outputs = [source_csv, table_csv, table_tex]
    for stem in (figure_stem, single_column_stem):
        for suffix in (".png", ".pdf", ".svg", ".tiff"):
            outputs.append(stem.with_suffix(suffix))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_assets(args.registry, args.gallery_root)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
