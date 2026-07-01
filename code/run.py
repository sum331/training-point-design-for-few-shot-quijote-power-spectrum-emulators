from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_translation_harness_loop import run_translation_harness


def _json_print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="z2 Quijote direct-CDM experiment runner")
    parser.add_argument("--config", dest="global_config", default=None)
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", dest="command_config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "show-config",
        parents=[config_parent],
        help="Validate config and print resolved paths",
    )

    build_parser = subparsers.add_parser(
        "build-splits",
        parents=[config_parent],
        help="Build disjoint seed/probe/pool/audit splits",
    )
    build_parser.add_argument("--force", action="store_true", help="Overwrite a same-label split output if needed")

    run_parser = subparsers.add_parser(
        "run-comparison",
        parents=[config_parent],
        help="Run fair direct-CDM baseline comparison",
    )
    run_parser.add_argument("--split-manifest", default=None, help="Use an existing split manifest JSON")
    run_parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume an interrupted comparison run from an existing run directory.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke",
        parents=[config_parent],
        help="Run a tiny synthetic direct-CDM smoke test",
    )
    smoke_parser.add_argument("--data-root", default=None, help="Optional smoke output root")

    translation_parser = subparsers.add_parser(
        "translation-harness",
        help="Run the manuscript translation harness/loop checks.",
    )
    translation_parser.add_argument("--source", type=Path, default=None, help="Chinese source Markdown.")
    translation_parser.add_argument("--candidate", type=Path, default=None, help="English candidate Markdown.")
    translation_parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACKAGE_ROOT / "docs" / "translation_harness_reports",
        help="Report output directory.",
    )
    translation_parser.add_argument("--seed", type=int, default=7)
    translation_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing exit code unless every harness case passes.",
    )

    args = parser.parse_args(argv)
    if args.command == "translation-harness":
        result = run_translation_harness(args)
        _json_print(result)
        if args.strict and result["pass_rate"] < 1.0:
            return 1
        return 0

    from z2quijote.config import load_config, make_smoke_config
    from z2quijote.experiment import run_fair_comparison
    from z2quijote.splits import build_split_bundle, load_split_bundle

    config_path = args.command_config or args.global_config or str(PACKAGE_ROOT / "config.yaml")
    config = load_config(config_path)

    if args.command == "show-config":
        _json_print(config.summary())
        return 0

    if args.command == "build-splits":
        bundle = build_split_bundle(config, force=bool(args.force))
        _json_print({"manifest_path": str(bundle.manifest_path), "arrays_path": str(bundle.arrays_path)})
        return 0

    if args.command == "run-comparison":
        split_bundle = load_split_bundle(Path(args.split_manifest)) if args.split_manifest else None
        result = run_fair_comparison(
            config,
            split_bundle=split_bundle,
            resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None,
        )
        artifacts = dict(result.summary.get("report_artifacts", {}))
        _json_print(
            {
                "run_dir": str(result.run_dir),
                "summary_path": str(result.summary_path),
                "report_manifest_path": result.summary.get("report_manifest_path"),
                "plot_dir": artifacts.get("plot_dir"),
                "plot_count": len(result.summary.get("plot_paths", [])),
            }
        )
        return 0

    if args.command == "smoke":
        smoke_config = make_smoke_config(config, data_root=Path(args.data_root) if args.data_root else None)
        bundle = build_split_bundle(smoke_config, force=True)
        result = run_fair_comparison(smoke_config, split_bundle=bundle)
        artifacts = dict(result.summary.get("report_artifacts", {}))
        _json_print(
            {
                "smoke": True,
                "manifest_path": str(bundle.manifest_path),
                "run_dir": str(result.run_dir),
                "summary_path": str(result.summary_path),
                "report_manifest_path": result.summary.get("report_manifest_path"),
                "plot_dir": artifacts.get("plot_dir"),
                "plot_count": len(result.summary.get("plot_paths", [])),
            }
        )
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
