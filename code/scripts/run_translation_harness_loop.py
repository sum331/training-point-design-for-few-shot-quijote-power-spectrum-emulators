"""Harness/loop checks for the Z2 Quijote manuscript translation workflow."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_SOURCE_CANDIDATES = (
    PROJECT_ROOT
    / "docs"
    / "paper_manuscript_20260701"
    / "chinese_source"
    / "z2quijote_manuscript_sections_1_2_cn_20260701.md",
    REPO_ROOT / "docs" / "z2quijote_manuscript_sections_1_2_cn_20260701.md",
)
DEFAULT_CANDIDATE_CANDIDATES = (
    PROJECT_ROOT / "docs" / "manuscript_word" / "z2quijote_full_manuscript_20260701.md",
    PROJECT_ROOT / "docs" / "paper_manuscript_20260701" / "z2quijote_english_manuscript_20260701.md",
    REPO_ROOT / "docs" / "z2quijote_manuscript_en_polished_20260701.md",
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "translation_harness_reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="Chinese source Markdown.")
    parser.add_argument("--candidate", type=Path, default=None, help="English candidate Markdown.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Report output directory.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing exit code unless every harness case passes.",
    )
    return parser.parse_args(argv)


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n".join(f"- {path}" for path in paths)
    raise FileNotFoundError(f"none of the candidate paths exists:\n{joined}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def display_math(text: str) -> list[str]:
    return [normalize_math(item) for item in re.findall(r"\\\[(.*?)\\\]", text, flags=re.S)]


def inline_math(text: str) -> list[str]:
    return [normalize_math(item) for item in re.findall(r"\\\((.*?)\\\)", text, flags=re.S)]


def normalize_math(text: str) -> str:
    # Manuscript candidates may add explicit equation numbers for journal
    # formatting.  These tags should not make formula-retention checks fail.
    text = re.sub(r"\\tag\{[^{}]*\}", "", text)
    return re.sub(r"\s+", " ", text.strip())


def headings(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.startswith("#")]


def markdown_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].strip().startswith(r"\["):
            start = i
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                i += 1
            if i < len(lines):
                i += 1
            body = "\n".join(lines[start:i])
            kind = "display_math"
        elif lines[i].startswith("#"):
            body = lines[i]
            kind = "heading"
            i += 1
        elif lines[i].startswith("|"):
            start = i
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            body = "\n".join(lines[start:i])
            kind = "table"
        elif lines[i].startswith(">"):
            start = i
            while i < len(lines) and lines[i].startswith(">"):
                i += 1
            body = "\n".join(lines[start:i])
            kind = "blockquote"
        else:
            start = i
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].startswith("#")
                and not lines[i].strip().startswith(r"\[")
            ):
                i += 1
            body = "\n".join(lines[start:i])
            kind = "paragraph"
        blocks.append(
            {
                "kind": kind,
                "chars": len(body),
                "inline_math": len(inline_math(body)),
                "display_math": len(display_math(body)),
                "preview": " ".join(body.split())[:160],
            }
        )
    return blocks


def chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def build_profile(source: Path, candidate: Path) -> dict[str, Any]:
    src = read_text(source)
    dst = read_text(candidate)
    src_display = display_math(src)
    dst_display = display_math(dst)
    src_inline = inline_math(src)
    dst_inline = inline_math(dst)
    missing_display = [item for item in src_display if item not in dst_display]
    src_inline_counts = Counter(src_inline)
    dst_inline_counts = Counter(dst_inline)
    missing_inline = []
    for item, count in src_inline_counts.items():
        if dst_inline_counts[item] < count:
            missing_inline.extend([item] * (count - dst_inline_counts[item]))
    key_numbers = [
        "0.019040",
        "0.014901",
        "21.74",
        "0.022609",
        "0.021251",
        "6.01",
        "0.013279",
        "0.013246",
        "0.017244",
        "0.014878",
        "0.021698",
        "0.016972",
        "0.019131",
        "0.014622",
    ]
    key_terms = [
        "Quijote",
        "KUN",
        "standard-geometry bias",
        "residual-anchor",
        "PCA-GP",
        "PPR",
        "variance-bias",
        "Delaunay",
        "Sobol",
        "overall",
    ]
    return {
        "source": str(source),
        "candidate": str(candidate),
        "source_chars": len(src),
        "candidate_chars": len(dst),
        "source_block_count": len(markdown_blocks(src)),
        "candidate_block_count": len(markdown_blocks(dst)),
        "source_headings": headings(src),
        "candidate_headings": headings(dst),
        "source_display_math_count": len(src_display),
        "candidate_display_math_count": len(dst_display),
        "missing_display_math_count": len(missing_display),
        "missing_display_math": missing_display[:20],
        "source_inline_math_count": len(src_inline),
        "candidate_inline_math_count": len(dst_inline),
        "missing_inline_math_count": len(missing_inline),
        "missing_inline_math": missing_inline[:20],
        "candidate_chinese_chars": chinese_chars(dst),
        "key_numbers_present": {item: item in dst for item in key_numbers},
        "key_terms_present": {item: item in dst for item in key_terms},
        "table_separator_count": dst.count("|---"),
        "figure_marker_count": dst.count("<!-- Fig") + dst.count("[[FIG:") + dst.count("Figure "),
    }


def metric_target(case_input: Any, config: Mapping[str, Any]) -> Any:
    profile = build_profile(Path(config["source"]), Path(config["candidate"]))
    metric = case_input["metric"]
    if metric == "display_math_count":
        return profile["candidate_display_math_count"]
    if metric == "display_math_missing":
        return profile["missing_display_math_count"]
    if metric == "inline_math_missing":
        return profile["missing_inline_math_count"]
    if metric == "chinese_residue":
        return profile["candidate_chinese_chars"]
    if metric == "heading_minimum":
        return len(profile["candidate_headings"]) >= min(6, max(1, len(profile["source_headings"]) // 3))
    if metric == "key_numbers":
        return all(profile["key_numbers_present"].values())
    if metric == "key_terms":
        return all(profile["key_terms_present"].values())
    if metric == "figures_present":
        return profile["figure_marker_count"] >= 7
    raise ValueError(f"unknown metric: {metric}")


def build_cases(profile: Mapping[str, Any]):
    from z2quijote.hloop import Case

    return [
        Case("display-math-count", {"metric": "display_math_count"}, profile["source_display_math_count"], {"type": "exact"}, ("formula",), 3.0),
        Case("display-math-missing", {"metric": "display_math_missing"}, 0, {"type": "exact"}, ("formula",), 5.0),
        Case("inline-math-missing", {"metric": "inline_math_missing"}, 0, {"type": "exact"}, ("formula",), 3.0),
        Case("no-chinese-residue", {"metric": "chinese_residue"}, 0, {"type": "exact"}, ("language",), 3.0),
        Case("heading-minimum", {"metric": "heading_minimum"}, True, {"type": "exact"}, ("structure",), 1.0),
        Case("key-numbers-present", {"metric": "key_numbers"}, True, {"type": "exact"}, ("numbers",), 3.0),
        Case("key-terms-present", {"metric": "key_terms"}, True, {"type": "exact"}, ("terminology",), 2.0),
        Case("figures-present", {"metric": "figures_present"}, True, {"type": "exact"}, ("structure",), 1.0),
    ]


def write_summary_md(profile: Mapping[str, Any], report: Any, path: Path) -> None:
    lines = [
        "# Z2 Translation Harness Report",
        "",
        f"- selected candidate: `{report.selected_candidate}`",
        f"- source: `{profile['source']}`",
        f"- candidate: `{profile['candidate']}`",
        f"- source chars: `{profile['source_chars']}`",
        f"- candidate chars: `{profile['candidate_chars']}`",
        f"- markdown blocks: `{profile['candidate_block_count']}` vs source `{profile['source_block_count']}`",
        f"- display math: `{profile['candidate_display_math_count']}/{profile['source_display_math_count']}`",
        f"- missing display math count: `{profile['missing_display_math_count']}`",
        f"- inline math count: `{profile['candidate_inline_math_count']}` vs source `{profile['source_inline_math_count']}`",
        f"- missing inline math count: `{profile['missing_inline_math_count']}`",
        f"- Chinese residue chars: `{profile['candidate_chinese_chars']}`",
        f"- figure marker count: `{profile['figure_marker_count']}`",
        "",
        "## Candidate Scores",
        "",
        "| candidate | utility | weighted score | pass rate | failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for evaluation in report.evaluations:
        summary = evaluation.summary
        lines.append(
            f"| {evaluation.candidate.name} | {evaluation.utility:.4f} | "
            f"{summary['weighted_score']:.4f} | {summary['pass_rate']:.2%} | "
            f"{summary['case_count'] - summary['pass_count']} |"
        )
    if profile["missing_display_math"]:
        lines += ["", "## First Missing Display Formulae", ""]
        for item in profile["missing_display_math"]:
            lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paragraph_audit(source: Path, candidate: Path, path: Path) -> None:
    src_blocks = markdown_blocks(read_text(source))
    dst_blocks = markdown_blocks(read_text(candidate))
    lines = [
        "# Z2 Translation Paragraph Audit",
        "",
        "This audit supports paragraph-by-paragraph review. A polished English manuscript may split or merge prose blocks, but headings, formulae, tables, and figure markers are checked by the harness.",
        "",
        f"- source blocks: `{len(src_blocks)}`",
        f"- candidate blocks: `{len(dst_blocks)}`",
        "",
        "## Source Block Summary",
        "",
        "| id | kind | inline math | display math | preview |",
        "|---:|---|---:|---:|---|",
    ]
    for idx, block in enumerate(src_blocks, 1):
        lines.append(f"| {idx} | {block['kind']} | {block['inline_math']} | {block['display_math']} | {block['preview'].replace('|', '/')} |")
    lines += ["", "## Candidate Block Summary", "", "| id | kind | inline math | display math | preview |", "|---:|---|---:|---:|---|"]
    for idx, block in enumerate(dst_blocks, 1):
        lines.append(f"| {idx} | {block['kind']} | {block['inline_math']} | {block['display_math']} | {block['preview'].replace('|', '/')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_translation_harness(args: argparse.Namespace) -> dict[str, Any]:
    from z2quijote.hloop import Candidate, Harness, LoopConfig, LoopController
    from z2quijote.hloop.report import write_json_report, write_markdown_report

    source = (args.source or first_existing(DEFAULT_SOURCE_CANDIDATES)).resolve()
    candidate = (args.candidate or first_existing(DEFAULT_CANDIDATE_CANDIDATES)).resolve()
    profile = build_profile(source, candidate)
    cases = build_cases(profile)
    candidates = [
        Candidate(
            name="z2_translation_candidate",
            config={"source": str(source), "candidate": str(candidate)},
            cost=0.0,
            risk=0.05,
        )
    ]
    controller = LoopController(Harness(seed=args.seed), LoopConfig(cost_weight=0.0, latency_weight=0.0, risk_weight=0.01))
    report = controller.run(cases, candidates, metric_target)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.out_dir / "z2_translation_harness_loop_report.json")
    write_markdown_report(report, args.out_dir / "z2_translation_harness_loop_report_raw.md")
    (args.out_dir / "z2_translation_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(profile, report, args.out_dir / "z2_translation_harness_loop_report.md")
    write_paragraph_audit(source, candidate, args.out_dir / "z2_translation_paragraph_audit.md")

    selected = next(e for e in report.evaluations if e.candidate.name == report.selected_candidate)
    return {
        "selected": report.selected_candidate,
        "pass_rate": selected.summary["pass_rate"],
        "weighted_score": selected.summary["weighted_score"],
        "display_math": f"{profile['candidate_display_math_count']}/{profile['source_display_math_count']}",
        "missing_display_math": profile["missing_display_math_count"],
        "inline_math": f"{profile['candidate_inline_math_count']}/{profile['source_inline_math_count']}",
        "missing_inline_math": profile["missing_inline_math_count"],
        "chinese_residue": profile["candidate_chinese_chars"],
        "source": str(source),
        "candidate": str(candidate),
        "report": str(args.out_dir / "z2_translation_harness_loop_report.md"),
        "json_report": str(args.out_dir / "z2_translation_harness_loop_report.json"),
        "paragraph_audit": str(args.out_dir / "z2_translation_paragraph_audit.md"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_translation_harness(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and result["pass_rate"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
