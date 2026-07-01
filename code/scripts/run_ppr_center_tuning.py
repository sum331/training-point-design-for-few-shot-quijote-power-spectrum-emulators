from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = PACKAGE_ROOT.parent
DEFAULT_BACKEND_ROOT = VERSIONS_ROOT / "R2_Multi-AL" / "R2-v2"
DEFAULT_CONFIG = PACKAGE_ROOT / "config_ppr.yaml"
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_BIAS_RUN_DIR = (
    DEFAULT_DATA_ROOT
    / "standard_geometry_bias"
    / "bias_m32768_n64_s900_tau016_k475_20260627T164634"
)
STATE_FILENAME = "adaptive_ppr_center_state.json"
STATUS_FILENAME = "adaptive_ppr_center_status.json"


RESULT_FIELDS = [
    "stage",
    "label",
    "blend_lambda",
    "overall_p68",
    "baseline_p68",
    "lhs_p68",
    "p68_improvement_fraction",
    "low_p68",
    "mid_p68",
    "focus_high_p68",
    "tail_p68",
    "high_p68",
    "relax_dir",
    "design_path",
    "validation_summary_path",
    "candidate_json",
]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    path: tuple[str, ...]
    mode: str
    initial_step: float
    min_step: float
    lower: float | None = None
    upper: float | None = None
    integer: bool = False


PARAMETERS: dict[str, ParameterSpec] = {
    "repulsion_length_scale": ParameterSpec(
        "repulsion_length_scale",
        ("lofi_design", "repulsion", "length_scale"),
        "additive",
        0.040,
        0.002,
        0.02,
        0.80,
    ),
    "adaptive_min_multiplier": ParameterSpec(
        "adaptive_min_multiplier",
        ("lofi_design", "repulsion", "adaptive_min_multiplier"),
        "additive",
        0.100,
        0.005,
        0.0,
        1.50,
    ),
    "nearest_neighbors": ParameterSpec(
        "nearest_neighbors",
        ("lofi_design", "repulsion", "nearest_neighbors"),
        "additive",
        8.0,
        1.0,
        4.0,
        63.0,
        True,
    ),
    "repulsion_force_cap": ParameterSpec(
        "repulsion_force_cap",
        ("lofi_design", "repulsion", "force_cap"),
        "additive",
        0.020,
        0.001,
        0.0,
        0.30,
    ),
    "adaptive_power": ParameterSpec(
        "adaptive_power",
        ("lofi_design", "repulsion", "adaptive_power"),
        "additive",
        0.500,
        0.020,
        0.10,
        12.0,
    ),
    "repulsion_softening": ParameterSpec(
        "repulsion_softening",
        ("lofi_design", "repulsion", "softening"),
        "additive",
        0.020,
        0.001,
        0.001,
        0.25,
    ),
    "potential_force_cap": ParameterSpec(
        "potential_force_cap",
        ("lofi_design", "potential_force_cap"),
        "additive",
        0.030,
        0.0025,
        0.0,
        0.25,
    ),
    "variance_power": ParameterSpec(
        "variance_power",
        ("potential_mapping", "variance_power"),
        "additive",
        0.250,
        0.010,
        0.05,
        5.0,
    ),
    "variance_gain": ParameterSpec(
        "variance_gain",
        ("potential_mapping", "variance_gain"),
        "additive",
        0.500,
        0.020,
        0.05,
        12.0,
    ),
    "damping": ParameterSpec(
        "damping",
        ("lofi_design", "damping"),
        "additive",
        0.040,
        0.005,
        0.50,
        0.995,
    ),
    "repulsion_strength": ParameterSpec(
        "repulsion_strength",
        ("lofi_design", "repulsion", "strength"),
        "additive",
        0.020,
        0.0005,
        0.0,
        0.20,
    ),
}


ADAPTIVE_LAYERS: tuple[dict[str, Any], ...] = (
    {
        "name": "repulsion_shape",
        "stage_prefix": "10_adaptive_ppr_repulsion_shape",
        "max_rounds": 1000,
        "round_relative_improvement_tol": 1.0e-3,
        "candidate_improvement_tol": 1.0e-8,
        "candidate_relative_improvement_tol": 1.0e-3,
        "first_round_candidate_relative_improvement_tol": 1.0e-2,
        "parameter_order": (
            "repulsion_length_scale",
            "adaptive_min_multiplier",
            "nearest_neighbors",
            "repulsion_force_cap",
            "adaptive_power",
            "repulsion_softening",
        ),
    },
    {
        "name": "potential_mapping",
        "stage_prefix": "20_adaptive_ppr_potential_mapping",
        "max_rounds": 1000,
        "round_relative_improvement_tol": 1.0e-3,
        "candidate_improvement_tol": 1.0e-8,
        "candidate_relative_improvement_tol": 1.0e-3,
        "first_round_candidate_relative_improvement_tol": 1.0e-2,
        "parameter_order": (
            "potential_force_cap",
            "variance_power",
            "variance_gain",
            "damping",
        ),
    },
    {
        "name": "repulsion_strength",
        "stage_prefix": "30_adaptive_ppr_repulsion_strength",
        "max_rounds": 1000,
        "round_relative_improvement_tol": 1.0e-3,
        "candidate_improvement_tol": 1.0e-8,
        "candidate_relative_improvement_tol": 1.0e-3,
        "first_round_candidate_relative_improvement_tol": 1.0e-2,
        "parameter_order": ("repulsion_strength",),
    },
    {
        "name": "joint_micro",
        "stage_prefix": "40_adaptive_ppr_joint_micro",
        "max_rounds": 1000,
        "round_relative_improvement_tol": 1.0e-3,
        "candidate_improvement_tol": 5.0e-9,
        "candidate_relative_improvement_tol": 1.0e-3,
        "first_round_candidate_relative_improvement_tol": 1.0e-2,
        "step_scale": 0.25,
        "parameter_order": (
            "repulsion_length_scale",
            "adaptive_min_multiplier",
            "nearest_neighbors",
            "repulsion_force_cap",
            "adaptive_power",
            "potential_force_cap",
            "variance_power",
            "variance_gain",
            "damping",
            "repulsion_strength",
        ),
    },
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return DEFAULT_DATA_ROOT / "ppr_tuning" / f"z2_ppr32_biasfield_center_tuning_{stamp}"


def _load_backend_tuning_module(backend_root: Path) -> Any:
    script = backend_root.resolve() / "scripts" / "tune_lofi_potential_design.py"
    if not script.exists():
        raise FileNotFoundError(f"PPR backend tuning script not found: {script}")
    spec = importlib.util.spec_from_file_location("z2_ppr_backend_tune", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PPR backend tuning script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_backend_src(backend_root: Path) -> None:
    src = backend_root.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _ensure_z2_src() -> None:
    src = PACKAGE_ROOT.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in RESULT_FIELDS}


def _as_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _parse_candidate_json(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("candidate_json", "")
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_overrides(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = _parse_candidate_json(row)
    overrides = payload.get("overrides")
    return copy.deepcopy(overrides) if isinstance(overrides, dict) else None


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    blend = row.get("blend_lambda", 1.0)
    try:
        blend_text = f"{float(blend):.8f}"
    except (TypeError, ValueError):
        blend_text = "1.00000000"
    return str(row.get("stage", "")), str(row.get("label", "")), blend_text


def _candidate_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["stage"]), str(record["label"]), "1.00000000"


def _is_baseline_reference(row: dict[str, Any]) -> bool:
    stage = str(row.get("stage", ""))
    label = str(row.get("label", ""))
    return "baseline" in label.lower() or stage == "00_reference_sobol_baseline"


def _center_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if _as_float(row, "overall_p68") is None:
            continue
        if _is_baseline_reference(row):
            continue
        if _parse_overrides(row) is None:
            continue
        out.append(row)
    return out


def _best_center_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = _center_candidates(rows)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["overall_p68"]))


def _best_any_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _as_float(row, "overall_p68") is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row["overall_p68"]))


def _deep_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _deep_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = payload
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _slug(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        text = f"{float(value):.6g}"
    except (TypeError, ValueError):
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return abs(float(left) - float(right)) <= 1.0e-10
    except (TypeError, ValueError):
        return left == right


def _clamp_value(value: float, spec: ParameterSpec, *, particles: int | None) -> float | int:
    lower = spec.lower
    upper = spec.upper
    if spec.name == "nearest_neighbors" and particles is not None:
        upper = min(float(max(1, int(particles) - 1)), float(upper or max(1, int(particles) - 1)))
    out = float(value)
    if lower is not None:
        out = max(float(lower), out)
    if upper is not None:
        out = min(float(upper), out)
    if spec.integer:
        return int(round(out))
    return round(out, 12)


def _candidate_values(
    *,
    center: float,
    spec: ParameterSpec,
    step: float,
    particles: int | None,
) -> list[float | int]:
    if spec.mode == "relative":
        step = min(0.90, max(0.0001, float(step)))
        raw = [
            center * (1.0 - step),
            center * (1.0 - 0.5 * step),
            center * (1.0 + 0.5 * step),
            center * (1.0 + step),
        ]
    elif spec.mode == "additive":
        step = max(0.0, float(step))
        raw = [
            center - step,
            center - 0.5 * step,
            center + 0.5 * step,
            center + step,
        ]
    else:
        raise ValueError(f"unknown parameter mode: {spec.mode}")
    values: list[float | int] = []
    for item in raw:
        value = _clamp_value(float(item), spec, particles=particles)
        if _same_value(value, center):
            continue
        if any(_same_value(value, old) for old in values):
            continue
        values.append(value)
    return values


def _base_overrides_from_config(config: Any) -> dict[str, Any]:
    return {
        "lofi_design": copy.deepcopy(config.section("lofi_design")),
        "potential_mapping": copy.deepcopy(config.section("potential_mapping")),
        "potential_weighting": copy.deepcopy(config.section("potential_weighting")),
    }


def _fill_default_values(config: Any, overrides: dict[str, Any], *, particles: int | None) -> dict[str, Any]:
    out = copy.deepcopy(overrides)
    base = _base_overrides_from_config(config)
    for section in ("potential_mapping", "potential_weighting"):
        merged = copy.deepcopy(base.get(section, {}))
        merged.update(copy.deepcopy(out.get(section, {})))
        out[section] = merged
    lofi = out.setdefault("lofi_design", {})
    default_lofi = base.get("lofi_design", {})
    for key, value in default_lofi.items():
        if key == "repulsion":
            continue
        lofi.setdefault(key, copy.deepcopy(value))
    if particles is not None:
        lofi["particles"] = int(particles)
    rep = lofi.setdefault("repulsion", {})
    default_rep = default_lofi.get("repulsion", {})
    if isinstance(default_rep, dict):
        for key, value in default_rep.items():
            rep.setdefault(key, copy.deepcopy(value))
    return out


def _path_looks_z2(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return "z2quijote" in text


def _assert_z2_source(path: Path, *, allow_external: bool, label: str) -> None:
    if allow_external:
        return
    if not _path_looks_z2(path.resolve()):
        raise ValueError(
            f"{label} must be under a z2quijote path. "
            f"Got {path}. Pass --allow-external-bootstrap only for an intentional migration."
        )


def _assert_current_bias_potential(path: Path, *, allow_legacy: bool) -> None:
    if allow_legacy:
        return
    summary_path = path.resolve().parent / "bias_field_potential_summary.json"
    if not summary_path.exists():
        raise ValueError(
            f"Cannot verify potential source because summary is missing: {summary_path}. "
            "Rebuild from --bias-run-dir or pass --allow-legacy-bias-potential for an explicit audit."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Potential summary is not valid JSON: {summary_path}") from exc
    source = summary.get("source_bias_run_dir")
    if source is not None and Path(str(source)).resolve() != DEFAULT_BIAS_RUN_DIR.resolve():
        raise ValueError(
            "Refusing to use a non-current z2 bias potential. "
            f"Expected source {DEFAULT_BIAS_RUN_DIR.resolve()}, got {Path(str(source)).resolve()}."
        )
    reference_points = int(summary.get("reference_points", 0))
    if reference_points < 32768:
        raise ValueError(
            f"Refusing to use an old low-reference bias potential: reference_points={reference_points}."
        )


def _load_bias_field(field_path: Path) -> dict[str, np.ndarray]:
    with np.load(field_path, allow_pickle=False) as payload:
        required = {"theta_unit", "bias_mean", "accepted_count"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"bias field is missing required arrays: {sorted(missing)}")
        return {
            "theta_unit": np.asarray(payload["theta_unit"], dtype=np.float64),
            "bias_mean": np.asarray(payload["bias_mean"], dtype=np.float64),
            "accepted_count": np.asarray(payload["accepted_count"], dtype=np.int64),
        }


def _metric_block(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p68": float(np.percentile(finite, 68)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _build_potential_from_bias_field(
    *,
    backend_root: Path,
    bias_run_dir: Path,
    out_dir: Path,
    min_count: int,
    high_confidence_count: int,
    neighbors: int,
    fallback_neighbors: int,
    length_scale: float,
    reference_size: int,
    force: bool,
) -> Path:
    _ensure_backend_src(backend_root)
    from r2_multi_al.bias_field_potential import BiasFieldScalarModel
    from r2_multi_al.potential import VariancePotential

    field_path = bias_run_dir.resolve() / "standard_geometry_bias_field.npz"
    if not field_path.exists():
        raise FileNotFoundError(f"standard geometry bias field not found: {field_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    potential_path = out_dir / "potential.pkl"
    summary_path = out_dir / "bias_field_potential_summary.json"
    if potential_path.exists() and summary_path.exists() and not force:
        return potential_path

    field = _load_bias_field(field_path)
    model = BiasFieldScalarModel(
        theta_unit=field["theta_unit"],
        bias=field["bias_mean"],
        accepted_count=field["accepted_count"],
        min_count=int(min_count),
        high_confidence_count=int(high_confidence_count),
        neighbors=int(neighbors),
        fallback_neighbors=int(fallback_neighbors),
        length_scale=float(length_scale),
    )
    refs = np.asarray(field["theta_unit"], dtype=np.float64)
    if refs.shape[0] > int(reference_size):
        rng = np.random.default_rng(20260624)
        idx = np.sort(rng.choice(refs.shape[0], size=int(reference_size), replace=False))
        refs = refs[idx]
    potential = VariancePotential(
        model,
        reference_points=refs.astype(np.float64),
        variance_floor=1.0e-12,
        low_percentile=5.0,
        high_percentile=95.0,
        rank_body_enabled=True,
        rank_body_quantile=0.68,
        rank_body_width=0.36,
        rank_body_floor=0.25,
        rank_body_mix=0.65,
    )
    with potential_path.open("wb") as handle:
        pickle.dump(potential, handle)

    support_values = model.scalar_variance(model.theta_unit)
    ref_values = potential.variance(refs, normalized=True)
    summary = {
        "status": "ok",
        "created_at": _now(),
        "source_bias_run_dir": str(bias_run_dir.resolve()),
        "source_bias_field": str(field_path),
        "potential_path": str(potential_path),
        "theta_dim": int(model.dim),
        "reference_points": int(refs.shape[0]),
        "support_points": int(model.theta_unit.shape[0]),
        "min_count": int(min_count),
        "high_confidence_count": int(high_confidence_count),
        "neighbors": int(neighbors),
        "fallback_neighbors": int(fallback_neighbors),
        "length_scale": float(length_scale),
        "support_bias": _metric_block(support_values),
        "reference_normalized_potential_variance": _metric_block(ref_values),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return potential_path


def _resolve_potential_path(args: argparse.Namespace, *, backend_root: Path, out_dir: Path) -> Path:
    if args.potential_path:
        path = Path(args.potential_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"potential path not found: {path}")
        _assert_z2_source(
            path,
            allow_external=bool(args.allow_external_bootstrap),
            label="potential_path",
        )
        _assert_current_bias_potential(path, allow_legacy=bool(args.allow_legacy_bias_potential))
        return path
    bias_run_dir = Path(args.bias_run_dir).resolve() if args.bias_run_dir else DEFAULT_BIAS_RUN_DIR.resolve()
    if not bias_run_dir.exists():
        raise ValueError(
            "Either --potential-path or a valid --bias-run-dir is required. "
            f"Default z2 bias run was not found: {bias_run_dir}"
        )
    _assert_z2_source(
        bias_run_dir,
        allow_external=bool(args.allow_external_bootstrap),
        label="bias_run_dir",
    )
    potential_out_dir = (
        Path(args.potential_out_dir).resolve()
        if args.potential_out_dir
        else out_dir / "bias_field_potential"
    )
    return _build_potential_from_bias_field(
        backend_root=backend_root,
        bias_run_dir=bias_run_dir,
        out_dir=potential_out_dir,
        min_count=int(args.bias_min_count),
        high_confidence_count=int(args.bias_high_confidence_count),
        neighbors=int(args.bias_neighbors),
        fallback_neighbors=int(args.bias_fallback_neighbors),
        length_scale=float(args.bias_length_scale),
        reference_size=int(args.bias_reference_size),
        force=bool(args.force_rebuild_potential),
    )


def _stage_name(layer: dict[str, Any], round_number: int, parameter: str, search_index: int, step: float) -> str:
    return (
        f"{layer['stage_prefix']}_r{int(round_number):02d}_"
        f"{parameter}_i{int(search_index):03d}_s{_slug(step)}"
    )


def _layer(state: dict[str, Any]) -> dict[str, Any] | None:
    index = int(state.get("layer_index", 0))
    if index < 0 or index >= len(ADAPTIVE_LAYERS):
        return None
    return ADAPTIVE_LAYERS[index]


def _parameter_order(layer: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in layer["parameter_order"])


def _initial_step(layer: dict[str, Any], spec: ParameterSpec, round_number: int) -> float:
    scale = float(layer.get("step_scale", 1.0))
    return max(float(spec.min_step), float(spec.initial_step) * scale * (0.5 ** max(0, int(round_number) - 1)))


def _candidate_relative_tol(layer: dict[str, Any], round_number: int) -> float:
    if int(round_number) <= 1:
        return float(layer.get("first_round_candidate_relative_improvement_tol", 1.0e-2))
    return float(layer.get("candidate_relative_improvement_tol", 1.0e-3))


def _round_policy(round_number: int) -> str:
    return "coarse_first_round" if int(round_number) <= 1 else "tight_convergence_round"


def _make_candidate_records(
    mod: Any,
    *,
    layer: dict[str, Any],
    parameter: str,
    center_row: dict[str, Any],
    center_overrides: dict[str, Any],
    step: float,
    round_number: int,
    search_index: int,
    particles: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    spec = PARAMETERS[parameter]
    center_value = _deep_get(center_overrides, spec.path)
    if center_value is None:
        raise ValueError(f"center row does not define parameter {parameter} at {spec.path}")
    values = _candidate_values(
        center=float(center_value),
        spec=spec,
        step=float(step),
        particles=particles,
    )
    stage = _stage_name(layer, round_number, parameter, search_index, step)
    records: list[dict[str, Any]] = []
    for value in values:
        overrides = copy.deepcopy(center_overrides)
        _deep_set(overrides, spec.path, value)
        label = f"{parameter}_{_slug(value)}"
        candidate = mod.Candidate(
            stage=stage,
            label=label,
            overrides=overrides,
            blend_lambdas=(1.0,),
        )
        records.append(
            {
                "candidate": candidate,
                "stage": stage,
                "label": label,
                "parameter": parameter,
                "path": list(spec.path),
                "value": value,
                "center_value": center_value,
                "center_row_key": list(_row_key(center_row)),
                "center_p68": float(center_row["overall_p68"]),
                "step": float(step),
                "mode": spec.mode,
                "overrides": overrides,
            }
        )
    return stage, records


def _serializable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        out.append(
            {
                "stage": record["stage"],
                "label": record["label"],
                "parameter": record["parameter"],
                "path": record["path"],
                "value": record["value"],
                "center_value": record["center_value"],
                "center_row_key": record["center_row_key"],
                "center_p68": record["center_p68"],
                "step": record["step"],
                "mode": record["mode"],
                "overrides": record["overrides"],
            }
        )
    return out


def _records_from_active(mod: Any, active: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in active.get("candidates", []):
        candidate = mod.Candidate(
            stage=str(item["stage"]),
            label=str(item["label"]),
            overrides=copy.deepcopy(item["overrides"]),
            blend_lambdas=(1.0,),
        )
        record = dict(item)
        record["candidate"] = candidate
        records.append(record)
    return records


def _row_from_source_center(source_center: dict[str, str]) -> dict[str, Any]:
    out = {field: source_center.get(field, "") for field in RESULT_FIELDS}
    out["stage"] = "00_reference_ppr_center"
    out["label"] = "source_best_ppr_center"
    payload = _parse_candidate_json(source_center)
    payload["source_stage"] = source_center.get("stage")
    payload["source_label"] = source_center.get("label")
    payload["reference_role"] = "center"
    out["candidate_json"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _result_row(out)


def _baseline_from_source(source_rows: list[dict[str, str]]) -> dict[str, Any] | None:
    baseline_rows = [
        row
        for row in source_rows
        if _as_float(row, "overall_p68") is not None
        and (str(row.get("stage", "")).startswith("00_") or "baseline" in str(row.get("label", "")).lower())
    ]
    if baseline_rows:
        source = min(baseline_rows, key=lambda row: float(row["overall_p68"]))
        out = {field: source.get(field, "") for field in RESULT_FIELDS}
        out["stage"] = "00_reference_sobol_baseline"
        out["label"] = "source_sobol_baseline"
        out["candidate_json"] = json.dumps(
            {"source_stage": source.get("stage"), "source_label": source.get("label")},
            ensure_ascii=False,
            sort_keys=True,
        )
        return _result_row(out)
    return None


def _best_source_center(source_rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = []
    for row in source_rows:
        if _as_float(row, "overall_p68") is None:
            continue
        if _is_baseline_reference(row):
            continue
        if _parse_overrides(row) is None:
            continue
        candidates.append(row)
    if not candidates:
        raise RuntimeError("No bootstrap PPR candidate with overrides was found.")
    return min(candidates, key=lambda row: float(row["overall_p68"]))


def _write_state(out_dir: Path, state: dict[str, Any]) -> None:
    (out_dir / STATE_FILENAME).write_text(
        json.dumps(_json_safe(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_state(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / STATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    center = _best_center_row(rows)
    if center is None:
        raise RuntimeError("Cannot initialize tuning state without a center row.")
    p68 = float(center["overall_p68"])
    return {
        "version": 1,
        "mode": "z2_ppr_adaptive_center",
        "status": "running",
        "created_at": _now(),
        "layer_index": 0,
        "round_number": 1,
        "parameter_index": 0,
        "search_index": 1,
        "next_step": None,
        "round_start_best_p68": p68,
        "layer_start_best_p68": p68,
        "active_search": None,
        "completed_layers": [],
        "adaptive_decisions": [],
        "adaptive_round_decisions": [],
    }


def _ensure_initial_rows(
    *,
    mod: Any,
    config: Any,
    rows: list[dict[str, Any]],
    results_path: Path,
    args: argparse.Namespace,
    potential_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    if _best_center_row(rows) is not None:
        return rows
    if args.bootstrap_results:
        bootstrap = Path(args.bootstrap_results).resolve()
        _assert_z2_source(
            bootstrap,
            allow_external=bool(args.allow_external_bootstrap),
            label="bootstrap_results",
        )
        source_rows = _read_rows(bootstrap)
        center = _row_from_source_center(_best_source_center(source_rows))
        rows.append(center)
        baseline = _baseline_from_source(source_rows)
        if baseline is not None:
            rows.append(baseline)
        _write_rows(results_path, rows)
        return rows
    if not args.init_from_config:
        raise RuntimeError(
            "No center row found. Provide --bootstrap-results or pass --init-from-config "
            "to validate the config-default PPR seed."
        )
    base_overrides = _fill_default_values(config, _base_overrides_from_config(config), particles=args.particles)
    candidate = mod.Candidate(
        stage="00_config_center",
        label="config_default_center",
        overrides=base_overrides,
        blend_lambdas=(1.0,),
    )
    record = {
        "candidate": candidate,
        "stage": candidate.stage,
        "label": candidate.label,
        "parameter": "config_center",
        "path": [],
        "value": "config_default",
        "center_value": "",
        "center_row_key": [],
        "center_p68": "",
        "step": "",
        "mode": "init",
        "overrides": base_overrides,
    }
    row = _run_one(
        mod=mod,
        config=config,
        out_dir=out_dir,
        record=record,
        potential_path=potential_path,
        validation_size=int(args.validation_size),
        validation_cache=args.validation_cache,
        k_count=args.k_count,
        pca_components=int(args.pca_components),
        gp_restarts=int(args.gp_restarts),
        relax_backend=str(args.relax_backend),
        relax_device=str(args.relax_device),
        relax_dtype=str(args.relax_dtype),
        relax_effective_neighbors=(
            None if int(args.relax_effective_neighbors) <= 0 else int(args.relax_effective_neighbors)
        ),
        rerun=bool(args.rerun),
    )
    rows.append(row)
    _write_rows(results_path, rows)
    return rows


def _create_active_search(
    *,
    mod: Any,
    config: Any,
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    particles: int | None,
) -> dict[str, Any]:
    layer = _layer(state)
    if layer is None:
        state["status"] = "complete"
        state["active_search"] = None
        return state
    parameter_order = _parameter_order(layer)
    parameter_index = int(state.get("parameter_index", 0))
    if parameter_index >= len(parameter_order):
        return _finish_round(rows=rows, state=state)

    center = _best_center_row(rows)
    if center is None:
        raise RuntimeError("No valid center row with overrides is available.")
    center_overrides = _parse_overrides(center)
    if center_overrides is None:
        raise RuntimeError("Best center row has no candidate overrides.")
    center_overrides = _fill_default_values(config, center_overrides, particles=particles)

    round_number = int(state.get("round_number", 1))
    parameter = parameter_order[parameter_index]
    spec = PARAMETERS[parameter]
    step = state.get("next_step")
    if step is None:
        step = _initial_step(layer, spec, round_number)
    step = max(float(spec.min_step), float(step))
    search_index = int(state.get("search_index", 1))
    stage, records = _make_candidate_records(
        mod,
        layer=layer,
        parameter=parameter,
        center_row=center,
        center_overrides=center_overrides,
        step=step,
        round_number=round_number,
        search_index=search_index,
        particles=particles,
    )
    state["active_search"] = {
        "stage": stage,
        "layer": layer["name"],
        "round_number": round_number,
        "parameter_order": list(parameter_order),
        "parameter": parameter,
        "round_policy": _round_policy(round_number),
        "step": step,
        "min_step": float(spec.min_step),
        "center_p68": float(center["overall_p68"]),
        "center_row_key": list(_row_key(center)),
        "center_parameters": _parameter_snapshot(center_overrides),
        "candidate_improvement_tol": float(layer["candidate_improvement_tol"]),
        "candidate_relative_improvement_tol": _candidate_relative_tol(layer, round_number),
        "candidates": _serializable_records(records),
    }
    return state


def _parameter_snapshot(overrides: dict[str, Any]) -> dict[str, Any]:
    snapshot = {}
    for name, spec in PARAMETERS.items():
        snapshot[name] = _deep_get(overrides, spec.path)
    return snapshot


def _finish_search(*, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    active = state.get("active_search") or {}
    stage = str(active.get("stage", ""))
    if not stage:
        return state
    layer = _layer(state)
    if layer is None:
        state["status"] = "complete"
        state["active_search"] = None
        return state

    stage_rows = [
        row
        for row in rows
        if str(row.get("stage", "")) == stage and _as_float(row, "overall_p68") is not None
    ]
    center_p68 = float(active.get("center_p68"))
    best_stage = _best_any_row(stage_rows)
    best_stage_p68 = float(best_stage["overall_p68"]) if best_stage is not None else float("inf")
    improvement_tol = float(active.get("candidate_improvement_tol", layer["candidate_improvement_tol"]))
    active_round = int(active.get("round_number", state.get("round_number", 1)))
    relative_tol = _candidate_relative_tol(layer, active_round)
    policy = str(active.get("round_policy") or _round_policy(active_round))
    parameter = str(active["parameter"])
    step = float(active["step"])
    min_step = float(active["min_step"])
    absolute_improvement = center_p68 - best_stage_p68
    relative_improvement = absolute_improvement / max(abs(center_p68), 1.0e-30)
    improved = best_stage_p68 + improvement_tol < center_p68
    relative_converged = improved and relative_tol > 0.0 and relative_improvement < relative_tol

    state["active_search"] = None
    state["search_index"] = int(state.get("search_index", 1)) + 1
    action: str
    if policy == "coarse_first_round":
        action = "advance_parameter_coarse"
        state["next_step"] = None
        state["parameter_index"] = int(state.get("parameter_index", 0)) + 1
        state["search_index"] = 1
    elif improved and not relative_converged:
        action = "continue_parameter"
        state["next_step"] = step
    elif relative_converged:
        action = "advance_parameter_relative_converged"
        state["next_step"] = None
        state["parameter_index"] = int(state.get("parameter_index", 0)) + 1
        state["search_index"] = 1
    elif step > min_step * 1.0001:
        action = "shrink_step"
        state["next_step"] = max(min_step, step * 0.5)
    else:
        action = "advance_parameter_min_step"
        state["next_step"] = None
        state["parameter_index"] = int(state.get("parameter_index", 0)) + 1
        state["search_index"] = 1

    state.setdefault("adaptive_decisions", []).append(
        {
            "stage": stage,
            "parameter": parameter,
            "center_p68": center_p68,
            "best_stage_p68": None if best_stage is None else best_stage_p68,
            "absolute_improvement": None if best_stage is None else absolute_improvement,
            "relative_improvement": None if best_stage is None else relative_improvement,
            "relative_improvement_tol": relative_tol,
            "step": step,
            "round_policy": policy,
            "action": action,
        }
    )

    current_layer = _layer(state)
    if current_layer is not None:
        parameter_order = _parameter_order(current_layer)
        if int(state.get("parameter_index", 0)) >= len(parameter_order):
            state = _finish_round(rows=rows, state=state)
    return state


def _finish_round(*, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    layer = _layer(state)
    best = _best_center_row(rows)
    best_p68 = float(best["overall_p68"]) if best is not None else float("inf")
    round_start = state.get("round_start_best_p68")
    if round_start is None:
        round_start = best_p68
    round_improvement = float(round_start) - best_p68
    round_relative_improvement = round_improvement / max(abs(float(round_start)), 1.0e-30)
    max_rounds = int(layer["max_rounds"]) if layer else 0
    round_number = int(state.get("round_number", 1))
    round_tol = float(layer.get("round_relative_improvement_tol", 0.0)) if layer else 0.0
    converged = round_tol > 0.0 and round_relative_improvement < round_tol
    if layer and not converged and round_number < max_rounds:
        state.setdefault("adaptive_round_decisions", []).append(
            {
                "layer": layer["name"],
                "round_number": round_number,
                "best_p68": best_p68,
                "round_improvement": round_improvement,
                "round_relative_improvement": round_relative_improvement,
                "round_relative_improvement_tol": round_tol,
                "action": "continue_layer_next_round",
            }
        )
        state["round_number"] = round_number + 1
        state["parameter_index"] = 0
        state["search_index"] = 1
        state["next_step"] = None
        state["round_start_best_p68"] = best_p68
        return state

    if layer:
        state.setdefault("completed_layers", []).append(
            {
                "layer": layer["name"],
                "rounds_completed": round_number,
                "best_p68": best_p68,
                "round_improvement": round_improvement,
                "round_relative_improvement": round_relative_improvement,
                "round_relative_improvement_tol": round_tol,
                "converged": bool(converged),
                "stopped_by_max_rounds": bool(not converged and round_number >= max_rounds),
            }
        )
    state["layer_index"] = int(state.get("layer_index", 0)) + 1
    state["round_number"] = 1
    state["parameter_index"] = 0
    state["search_index"] = 1
    state["next_step"] = None
    state["round_start_best_p68"] = best_p68
    state["layer_start_best_p68"] = best_p68
    if _layer(state) is None:
        state["status"] = "complete"
    return state


def _run_one(
    *,
    mod: Any,
    config: Any,
    out_dir: Path,
    record: dict[str, Any],
    potential_path: Path,
    validation_size: int,
    validation_cache: str | None,
    k_count: int | None,
    pca_components: int,
    gp_restarts: int,
    relax_backend: str,
    relax_device: str,
    relax_dtype: str,
    relax_effective_neighbors: int | None,
    rerun: bool,
) -> dict[str, Any]:
    candidate = record["candidate"]
    candidate_config = mod._config_with_overrides(config, candidate.overrides)
    candidate_dir = out_dir / candidate.stage / candidate.label
    relax_dir = candidate_dir / "relax"
    relax_summary = relax_dir / "lofi_relaxation_summary.json"
    if rerun or not relax_summary.exists():
        backend = str(relax_backend or "r2").strip().lower()
        used_fast = False
        if backend in {"torch", "z2_torch", "gpu", "cuda"}:
            try:
                _ensure_z2_src()
                from z2quijote.ppr_fast_relaxation import run_fast_lofi_relaxation_from_bias_potential

                run_fast_lofi_relaxation_from_bias_potential(
                    candidate_config,
                    potential_path,
                    relax_dir,
                    device=relax_device,
                    dtype=relax_dtype,
                    effective_neighbors=relax_effective_neighbors,
                )
                used_fast = True
            except Exception:
                raise
        if not used_fast:
            mod.run_lofi_relaxation_from_potential(candidate_config, potential_path, relax_dir)
    source_design = relax_dir / "lofi_design.npz"
    blend_dir = candidate_dir / "blend_lambda_1"
    design_path = blend_dir / "lofi_design.npz"
    validation_dir = blend_dir / "quijote_validation"
    summary_path = validation_dir / "quijote_validation_comparison_summary.json"
    if rerun or not design_path.exists():
        mod._save_blend_design(
            source_design=source_design,
            output_design=design_path,
            blend_lambda=1.0,
            config=candidate_config,
        )
    if rerun or not summary_path.exists():
        mod.validate_lofi_design_with_quijote_gp(
            candidate_config,
            design_path,
            validation_dir,
            validation_size=int(validation_size),
            validation_cache=validation_cache,
            k_count_override=k_count,
            pca_components=int(pca_components),
            gp_restarts=int(gp_restarts),
            save_predictions=True,
        )
    if hasattr(mod, "_cleanup_validation_predictions"):
        mod._cleanup_validation_predictions(validation_dir)
    row = mod._extract_validation_row(
        candidate=candidate,
        blend_lambda=1.0,
        relax_dir=relax_dir,
        design_path=design_path,
        summary_path=summary_path,
    )
    row["candidate_json"] = json.dumps(
        {
            "stage": candidate.stage,
            "label": candidate.label,
            "overrides": candidate.overrides,
            "adaptive_ppr_center": {
                "parameter": record.get("parameter"),
                "value": record.get("value"),
                "center_value": record.get("center_value"),
                "center_p68": record.get("center_p68"),
                "center_row_key": record.get("center_row_key"),
                "step": record.get("step"),
                "mode": record.get("mode"),
                "path": record.get("path"),
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _result_row(row)


def _write_best_files(out_dir: Path, rows: list[dict[str, Any]], *, potential_path: Path, config_path: Path) -> None:
    best = _best_center_row(rows)
    if best is None:
        return
    payload = {
        "status": "complete" if (out_dir / STATE_FILENAME).exists() else "running",
        "selection_metric": "overall p68 relative error",
        "best_row": best,
        "overrides": _parse_overrides(best),
        "design_path": best.get("design_path"),
        "validation_summary_path": best.get("validation_summary_path"),
        "potential_path": str(potential_path),
        "config_path": str(config_path),
        "results_path": str(out_dir / "results.csv"),
        "state_path": str(out_dir / STATE_FILENAME),
    }
    for name in ("best_parameters.json", "best_ppr_parameters.json"):
        (out_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_audit_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if _as_float(row, "overall_p68") is not None]
    if not valid:
        return
    sorted_rows = sorted(valid, key=lambda row: float(row["overall_p68"]))
    best = sorted_rows[0]
    lines = [
        "# Z2 PPR Center Tuning",
        "",
        "Metric: Quijote validation overall p68 relative error only.",
        "",
        "Best so far:",
        "",
        f"- stage: `{best.get('stage')}`",
        f"- label: `{best.get('label')}`",
        f"- overall_p68: `{float(best['overall_p68']):.12g}`",
        f"- design_path: `{best.get('design_path', '')}`",
        "",
        "Top rows:",
        "",
        "| rank | stage | label | p68 | design |",
        "|---:|---|---|---:|---|",
    ]
    for rank, row in enumerate(sorted_rows[:12], start=1):
        lines.append(
            f"| {rank} | `{row.get('stage')}` | `{row.get('label')}` | "
            f"{float(row['overall_p68']):.8g} | `{row.get('design_path', '')}` |"
        )
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status(
    out_dir: Path,
    *,
    status: str,
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    started_at: float | None,
    current: dict[str, Any] | None = None,
    potential_path: Path | None = None,
) -> None:
    elapsed = None if started_at is None else max(0.0, time.time() - started_at)
    active = state.get("active_search") or {}
    active_records = active.get("candidates", [])
    existing = {_row_key(row) for row in rows}
    pending_in_active = [
        record
        for record in active_records
        if (str(record.get("stage")), str(record.get("label")), "1.00000000") not in existing
    ]
    payload = {
        "status": status,
        "updated_at": _now(),
        "elapsed_seconds": elapsed,
        "state_path": str(out_dir / STATE_FILENAME),
        "results_path": str(out_dir / "results.csv"),
        "potential_path": None if potential_path is None else str(potential_path),
        "best_center": _best_center_row(rows),
        "best_including_baseline": _best_any_row(rows),
        "state_summary": {
            "layer_index": state.get("layer_index"),
            "layer": None if _layer(state) is None else _layer(state).get("name"),
            "round_number": state.get("round_number"),
            "parameter_index": state.get("parameter_index"),
            "search_index": state.get("search_index"),
            "next_step": state.get("next_step"),
            "active_stage": active.get("stage"),
            "active_parameter": active.get("parameter"),
            "active_candidates": len(active_records),
            "active_pending": len(pending_in_active),
        },
        "current": current,
    }
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    backend_root = Path(args.backend_root).resolve()
    mod = _load_backend_tuning_module(backend_root)
    config_path = Path(args.config).resolve()
    config = mod.load_config(config_path)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.particles is not None:
        os.environ["R2_LOFI_PARTICLES"] = str(int(args.particles))

    potential_path = _resolve_potential_path(args, backend_root=backend_root, out_dir=out_dir)
    if bool(args.build_potential_only):
        print(
            json.dumps(
                {
                    "status": "potential_ready",
                    "out_dir": str(out_dir),
                    "potential_path": str(potential_path),
                    "bias_run_dir": str(Path(args.bias_run_dir).resolve() if args.bias_run_dir else DEFAULT_BIAS_RUN_DIR),
                    "bias_reference_size": int(args.bias_reference_size),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    results_path = out_dir / "results.csv"
    rows: list[dict[str, Any]] = [_result_row(dict(row)) for row in _read_rows(results_path)]

    if args.reset_state:
        state_path = out_dir / STATE_FILENAME
        if state_path.exists():
            state_path.unlink()

    rows = _ensure_initial_rows(
        mod=mod,
        config=config,
        rows=rows,
        results_path=results_path,
        args=args,
        potential_path=potential_path,
        out_dir=out_dir,
    )
    state = _load_state(out_dir)
    if state is None:
        state = _initial_state(rows)
        _write_state(out_dir, state)

    if args.dry_run:
        if not state.get("active_search"):
            state = _create_active_search(
                mod=mod,
                config=config,
                rows=rows,
                state=state,
                particles=args.particles,
            )
        records = _records_from_active(mod, state.get("active_search") or {})
        existing = {_row_key(row) for row in rows}
        pending = [record for record in records if _candidate_key(record) not in existing]
        _write_status(
            out_dir,
            status="dry_run",
            rows=rows,
            state=state,
            started_at=None,
            potential_path=potential_path,
        )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "out_dir": str(out_dir),
                    "config_path": str(config_path),
                    "potential_path": str(potential_path),
                    "best_center_p68": float(_best_center_row(rows)["overall_p68"]),
                    "active_search": state.get("active_search", {}),
                    "pending_preview": [
                        {
                            "stage": record["stage"],
                            "label": record["label"],
                            "parameter": record["parameter"],
                            "value": record["value"],
                            "center_value": record["center_value"],
                        }
                        for record in pending[:8]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    started_at = time.time()
    started = 0
    max_new = int(args.max_new_candidates)
    unlimited = max_new <= 0
    while unlimited or started < max_new:
        if state.get("status") == "complete":
            break
        if not state.get("active_search"):
            state = _create_active_search(
                mod=mod,
                config=config,
                rows=rows,
                state=state,
                particles=args.particles,
            )
            _write_state(out_dir, state)
            if state.get("status") == "complete":
                break

        active = state.get("active_search") or {}
        records = _records_from_active(mod, active)
        existing = {_row_key(row) for row in rows}
        pending = [record for record in records if _candidate_key(record) not in existing]
        if not pending:
            state = _finish_search(rows=rows, state=state)
            _write_state(out_dir, state)
            continue

        record = pending[0]
        current = {
            "stage": record["stage"],
            "label": record["label"],
            "parameter": record["parameter"],
            "value": record["value"],
            "center_value": record["center_value"],
            "started_at": _now(),
        }
        _write_status(
            out_dir,
            status="running",
            rows=rows,
            state=state,
            started_at=started_at,
            current=current,
            potential_path=potential_path,
        )
        row = _run_one(
            mod=mod,
            config=config,
            out_dir=out_dir,
            record=record,
            potential_path=potential_path,
            validation_size=int(args.validation_size),
            validation_cache=args.validation_cache,
            k_count=args.k_count,
            pca_components=int(args.pca_components),
            gp_restarts=int(args.gp_restarts),
            relax_backend=str(args.relax_backend),
            relax_device=str(args.relax_device),
            relax_dtype=str(args.relax_dtype),
            relax_effective_neighbors=(
                None if int(args.relax_effective_neighbors) <= 0 else int(args.relax_effective_neighbors)
            ),
            rerun=bool(args.rerun),
        )
        rows.append(row)
        _write_rows(results_path, rows)
        _write_audit_report(out_dir, rows)
        _write_best_files(out_dir, rows, potential_path=potential_path, config_path=config_path)
        started += 1
        print(
            json.dumps(
                {
                    "status": "candidate_validated",
                    "started_candidates": started,
                    "stage": row.get("stage"),
                    "label": row.get("label"),
                    "overall_p68": row.get("overall_p68"),
                    "best_center_p68": _best_center_row(rows).get("overall_p68"),
                    "results_path": str(results_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        _write_status(
            out_dir,
            status="running",
            rows=rows,
            state=state,
            started_at=started_at,
            potential_path=potential_path,
        )

    final_status = "complete" if state.get("status") == "complete" else "paused_max_new_candidates"
    _write_state(out_dir, state)
    _write_audit_report(out_dir, rows)
    _write_best_files(out_dir, rows, potential_path=potential_path, config_path=config_path)
    _write_status(
        out_dir,
        status=final_status,
        rows=rows,
        state=state,
        started_at=started_at,
        potential_path=potential_path,
    )
    print(
        json.dumps(
            {
                "status": final_status,
                "started_candidates": started,
                "out_dir": str(out_dir),
                "results_path": str(results_path),
                "state_path": str(out_dir / STATE_FILENAME),
                "status_path": str(out_dir / STATUS_FILENAME),
                "best_parameters_path": str(out_dir / "best_parameters.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive center-coordinate tuning for z2 bias-field PPR seeds."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backend-root", default=str(DEFAULT_BACKEND_ROOT))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-results", default=None)
    parser.add_argument("--allow-external-bootstrap", action="store_true")
    parser.add_argument("--init-from-config", action="store_true")
    parser.add_argument("--potential-path", default=None)
    parser.add_argument("--bias-run-dir", default=str(DEFAULT_BIAS_RUN_DIR))
    parser.add_argument("--potential-out-dir", default=None)
    parser.add_argument("--build-potential-only", action="store_true")
    parser.add_argument("--force-rebuild-potential", action="store_true")
    parser.add_argument("--allow-legacy-bias-potential", action="store_true")
    parser.add_argument("--bias-min-count", type=int, default=10)
    parser.add_argument("--bias-high-confidence-count", type=int, default=20)
    parser.add_argument("--bias-neighbors", type=int, default=96)
    parser.add_argument("--bias-fallback-neighbors", type=int, default=160)
    parser.add_argument("--bias-length-scale", type=float, default=0.18)
    parser.add_argument("--bias-reference-size", type=int, default=32768)
    parser.add_argument(
        "--max-new-candidates",
        type=int,
        default=0,
        help="Maximum new candidates to evaluate in this invocation. Use 0 for no candidate-count limit.",
    )
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--validation-cache", type=str, default=None)
    parser.add_argument("--k-count", type=int, default=475)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--gp-restarts", type=int, default=0)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument(
        "--relax-backend",
        choices=("auto", "torch", "r2"),
        default="r2",
        help="PPR motion backend. r2 uses the current CPU cKDTree relaxation; torch is an explicit experiment.",
    )
    parser.add_argument("--relax-device", default="auto")
    parser.add_argument("--relax-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--relax-effective-neighbors",
        type=int,
        default=0,
        help="Torch KNN neighbors for bias interpolation. Use 0 for max(neighbors, fallback_neighbors).",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
