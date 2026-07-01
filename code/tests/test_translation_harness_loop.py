from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import main as run_main
from scripts.run_translation_harness_loop import run_translation_harness


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source_cn.md"
    candidate = tmp_path / "candidate_en.md"
    formula = r"\[ r_Q(\theta,k)=\log P_Q^{\rm nl}(\theta,k)-\log P_{\rm anchor}^{\rm CDM,nl}(\theta,k) \]"
    inline = r"\(p68\)"
    source.write_text(
        "\n".join(
            [
                "# 中文源稿",
                "",
                f"中文段落保留公式 {inline}。",
                "",
                formula,
                "",
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        "\n".join(
            [
                "# English candidate",
                "",
                "Quijote KUN standard-geometry bias residual-anchor PCA-GP PPR variance-bias Delaunay Sobol overall.",
                "",
                f"The metric {inline} is retained.",
                "",
                formula,
                "",
                "0.019040 0.014901 21.74 0.022609 0.021251 6.01 0.013279 0.013246.",
                "0.017244 0.014878 0.021698 0.016972 0.019131 0.014622.",
                "Figure 1. Figure 2. Figure 3. Figure 4. Figure 5. Figure 6. Figure 7.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source, candidate


def test_translation_harness_loop_runs_without_lab_dependency(tmp_path: Path) -> None:
    source, candidate = _write_pair(tmp_path)
    out_dir = tmp_path / "reports"
    args = Namespace(source=source, candidate=candidate, out_dir=out_dir, seed=3, strict=True)

    result = run_translation_harness(args)

    assert result["pass_rate"] == 1.0
    assert result["missing_display_math"] == 0
    assert result["missing_inline_math"] == 0
    assert Path(result["report"]).exists()
    assert Path(result["json_report"]).exists()
    assert Path(result["paragraph_audit"]).exists()


def test_translation_harness_ignores_equation_tags(tmp_path: Path) -> None:
    source, candidate = _write_pair(tmp_path)
    raw = candidate.read_text(encoding="utf-8")
    candidate.write_text(raw.replace(r"\]", "\n\\tag{1}\n\\]", 1), encoding="utf-8")
    out_dir = tmp_path / "reports"
    args = Namespace(source=source, candidate=candidate, out_dir=out_dir, seed=3, strict=True)

    result = run_translation_harness(args)

    assert result["missing_display_math"] == 0


def test_run_py_translation_harness_subcommand(tmp_path: Path) -> None:
    source, candidate = _write_pair(tmp_path)
    out_dir = tmp_path / "reports"

    exit_code = run_main(
        [
            "translation-harness",
            "--source",
            str(source),
            "--candidate",
            str(candidate),
            "--out-dir",
            str(out_dir),
            "--strict",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "z2_translation_harness_loop_report.md").exists()
