"""Helpers for per-run artifact layout.

New runs use the following structure under each run root:
- <run>/results/: final outputs, reports, plots, evaluation payloads
- <run>/process/: streaming logs, runtime events, intermediate process payloads

Readers remain backward compatible with the legacy flat run layout by falling back
 to <run>/<relative_path> when the new location does not exist.
"""

from __future__ import annotations

from pathlib import Path

RESULTS_DIRNAME = "results"
PROCESS_DIRNAME = "process"


def _run_root(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve()


def ensure_run_artifact_layout(run_dir: str | Path) -> tuple[Path, Path]:
    root = _run_root(run_dir)
    results_dir = root / RESULTS_DIRNAME
    process_dir = root / PROCESS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, process_dir


def run_results_dir(run_dir: str | Path, *, create: bool = False) -> Path:
    root = _run_root(run_dir)
    results_dir = root / RESULTS_DIRNAME
    if create:
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    return results_dir if results_dir.exists() else root


def run_process_dir(run_dir: str | Path, *, create: bool = False) -> Path:
    root = _run_root(run_dir)
    process_dir = root / PROCESS_DIRNAME
    if create:
        process_dir.mkdir(parents=True, exist_ok=True)
        return process_dir
    return process_dir if process_dir.exists() else root


def _run_artifact_path(
    run_dir: str | Path,
    *relative_parts: str | Path,
    kind: str,
    create: bool = False,
) -> Path:
    root = _run_root(run_dir)
    relative = Path(*[str(part) for part in relative_parts]) if relative_parts else Path()
    if kind == "results":
        modern_base = run_results_dir(root, create=create)
    elif kind == "process":
        modern_base = run_process_dir(root, create=create)
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")

    modern_path = (modern_base / relative).resolve()
    if create or modern_path.exists():
        return modern_path

    legacy_path = (root / relative).resolve()
    if legacy_path.exists():
        return legacy_path
    return modern_path


def run_results_path(run_dir: str | Path, *relative_parts: str | Path, create: bool = False) -> Path:
    return _run_artifact_path(run_dir, *relative_parts, kind="results", create=create)


def run_process_path(run_dir: str | Path, *relative_parts: str | Path, create: bool = False) -> Path:
    return _run_artifact_path(run_dir, *relative_parts, kind="process", create=create)


def run_results_subdir(run_dir: str | Path, *relative_parts: str | Path, create: bool = False) -> Path:
    path = run_results_path(run_dir, *relative_parts, create=create)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def run_process_subdir(run_dir: str | Path, *relative_parts: str | Path, create: bool = False) -> Path:
    path = run_process_path(run_dir, *relative_parts, create=create)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def derive_run_dir_from_artifact(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_dir() and resolved.name in {RESULTS_DIRNAME, PROCESS_DIRNAME}:
        return resolved.parent
    parent = resolved.parent
    if parent.name in {RESULTS_DIRNAME, PROCESS_DIRNAME}:
        return parent.parent
    return parent
