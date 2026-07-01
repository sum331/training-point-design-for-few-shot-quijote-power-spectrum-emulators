from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACKAGE_ROOT / "src"
for path in (PACKAGE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from z2quijote.config import load_config
from z2quijote.reporting import generate_run_report
from z2quijote.splits import load_split_bundle


def _load_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the full z2 run report package.")
    parser.add_argument("--run-dir", type=Path, required=True, help="z2 run directory to package.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config.yaml",
        help="z2 config used to reproduce the validation payloads.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional explicit fair-comparison summary JSON path.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional split manifest JSON path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    config = load_config(args.config)
    summary_path = Path(args.summary).resolve() if args.summary is not None else None
    summary = _load_json(summary_path) if summary_path is not None else None
    split_bundle = load_split_bundle(Path(args.split_manifest).resolve()) if args.split_manifest is not None else None
    manifest = generate_run_report(
        config=config,
        run_dir=run_dir,
        summary=summary,
        summary_path=summary_path,
        split_bundle=split_bundle,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
