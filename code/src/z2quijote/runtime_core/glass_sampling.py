"""Standalone glass-design generator for high-dimensional parameter spaces.

This module adapts the cosmological "glass" idea to the emulator's parameter
hypercube. The classical glass construction starts from a Poisson realization
in a periodic box and evolves particles under reversed gravity until the net
force becomes small. For design generation inside an 8D parameter cube we use
the same core idea:

1. Work on the unit hypercube either as a flat torus (periodic boundary
   conditions) or as a bounded cube with reflective hard walls.
2. Start from a scrambled Sobol or random point set.
3. Relax the points under overdamped pairwise repulsion.

The important dimensional point is that the 3D inverse-square law should not be
copied literally into 8D if we want the force law to match the d-dimensional
Coulomb/Green's-function scaling. In d spatial dimensions the fundamental
solution of the Laplacian scales like r^(2-d), so the repulsive force magnitude
scales like r^(1-d). In 8D that means:

    potential ~ 1 / r^6
    force     ~ 1 / r^7

We therefore default to a dimension-aware Riesz/Coulomb exponent. A heuristic
"inverse_square" mode is also exposed for side-by-side experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from z2quijote.runtime_core.config import (
    ValidationRuntimeConfig,
    build_default_config,
    denormalize_theta_batch,
    theta_bounds_as_array,
)
from z2quijote.runtime_core.sampling import generate_unit_sobol_samples


@dataclass(slots=True)
class GlassDesignConfig:
    n_points: int = 128
    theta_dim: int = 8
    init_mode: str = "sobol"
    random_seed: int = 20260407
    boundary_mode: str = "periodic"
    force_model: str = "dimension_aware_coulomb"
    riesz_exponent: float | None = None
    softening: float = 1.0e-3
    max_iterations: int = 4000
    min_iterations: int = 512
    initial_step_size: float = 0.05
    max_step_size: float = 0.10
    min_step_size: float = 1.0e-6
    step_growth: float = 1.02
    stable_rel_energy_tol: float = 1.0e-9
    stable_rel_metric_tol: float = 1.0e-7
    stable_iteration_window: int = 64
    max_backtracking_steps: int = 24
    history_stride: int = 20
    output_path: str | None = None

    def __post_init__(self) -> None:
        self.n_points = max(2, int(self.n_points))
        self.theta_dim = max(1, int(self.theta_dim))
        self.init_mode = str(self.init_mode).strip().lower() or "sobol"
        if self.init_mode not in {"sobol", "random"}:
            raise ValueError(
                "init_mode must be one of {'sobol', 'random'}, "
                f"got {self.init_mode!r}."
            )
        self.boundary_mode = str(self.boundary_mode).strip().lower() or "periodic"
        if self.boundary_mode not in {"periodic", "reflective"}:
            raise ValueError(
                "glass sampling supports only {'periodic', 'reflective'} boundaries, "
                f"got {self.boundary_mode!r}."
            )
        self.force_model = str(self.force_model).strip().lower() or "dimension_aware_coulomb"
        if self.force_model not in {"dimension_aware_coulomb", "inverse_square", "riesz"}:
            raise ValueError(
                "force_model must be one of {'dimension_aware_coulomb', 'inverse_square', 'riesz'}, "
                f"got {self.force_model!r}."
            )
        self.softening = float(max(1.0e-12, self.softening))
        self.max_iterations = max(1, int(self.max_iterations))
        self.min_iterations = int(min(max(0, int(self.min_iterations)), self.max_iterations))
        self.initial_step_size = float(max(1.0e-6, self.initial_step_size))
        self.max_step_size = float(max(self.initial_step_size, self.max_step_size))
        self.min_step_size = float(max(1.0e-12, min(self.min_step_size, self.initial_step_size)))
        self.step_growth = float(max(1.0, self.step_growth))
        self.stable_rel_energy_tol = float(max(0.0, self.stable_rel_energy_tol))
        self.stable_rel_metric_tol = float(max(0.0, self.stable_rel_metric_tol))
        self.stable_iteration_window = max(4, int(self.stable_iteration_window))
        self.max_backtracking_steps = max(1, int(self.max_backtracking_steps))
        self.history_stride = max(1, int(self.history_stride))


@dataclass(slots=True)
class GlassDesignResult:
    unit_points: np.ndarray
    raw_points: np.ndarray
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _resolve_riesz_exponent(config: GlassDesignConfig) -> float:
    if config.force_model == "dimension_aware_coulomb":
        if config.theta_dim <= 2:
            raise ValueError(
                "dimension_aware_coulomb requires theta_dim >= 3 because the "
                "d=2 Coulomb case is logarithmic."
            )
        return float(config.theta_dim - 2)
    if config.force_model == "inverse_square":
        return 1.0
    if config.riesz_exponent is None:
        raise ValueError("force_model='riesz' requires riesz_exponent to be provided.")
    return float(max(1.0e-8, config.riesz_exponent))


def _generate_initial_unit_points(config: GlassDesignConfig) -> np.ndarray:
    if config.init_mode == "sobol":
        return generate_unit_sobol_samples(
            theta_dim=int(config.theta_dim),
            sample_size=int(config.n_points),
            random_seed=int(config.random_seed),
            scramble=True,
        )
    rng = np.random.default_rng(int(config.random_seed))
    return rng.random((int(config.n_points), int(config.theta_dim)), dtype=np.float64)


def _minimum_image_displacements(points: np.ndarray) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64)[:, None, :] - np.asarray(points, dtype=np.float64)[None, :, :]
    return delta - np.rint(delta)


def _bounded_displacements(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)[:, None, :] - np.asarray(points, dtype=np.float64)[None, :, :]


def compute_periodic_riesz_energy(
    points: np.ndarray,
    *,
    riesz_exponent: float,
    softening: float,
) -> float:
    delta = _minimum_image_displacements(points)
    dist2 = np.sum(delta * delta, axis=2) + float(softening) ** 2
    np.fill_diagonal(dist2, np.inf)
    upper = np.triu_indices(dist2.shape[0], k=1)
    return float(np.sum(np.power(dist2[upper], -0.5 * float(riesz_exponent))))


def compute_bounded_riesz_energy(
    points: np.ndarray,
    *,
    riesz_exponent: float,
    softening: float,
) -> float:
    delta = _bounded_displacements(points)
    dist2 = np.sum(delta * delta, axis=2) + float(softening) ** 2
    np.fill_diagonal(dist2, np.inf)
    upper = np.triu_indices(dist2.shape[0], k=1)
    return float(np.sum(np.power(dist2[upper], -0.5 * float(riesz_exponent))))


def compute_periodic_repulsive_forces(
    points: np.ndarray,
    *,
    riesz_exponent: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=np.float64)
    delta = _minimum_image_displacements(points_arr)
    dist2 = np.sum(delta * delta, axis=2) + float(softening) ** 2
    np.fill_diagonal(dist2, np.inf)

    # For U = sum_{i<j} 1 / r^s, the descent direction is:
    #   F_i = s * sum_j (x_i - x_j) / r^{s+2}
    s = float(riesz_exponent)
    pair_scale = s * np.power(dist2, -0.5 * (s + 2.0))
    pair_scale[~np.isfinite(pair_scale)] = 0.0
    forces = np.sum(delta * pair_scale[:, :, None], axis=1).astype(np.float64)
    force_norm = np.linalg.norm(forces, axis=1)
    return forces, force_norm.astype(np.float64)


def compute_bounded_repulsive_forces(
    points: np.ndarray,
    *,
    riesz_exponent: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=np.float64)
    delta = _bounded_displacements(points_arr)
    dist2 = np.sum(delta * delta, axis=2) + float(softening) ** 2
    np.fill_diagonal(dist2, np.inf)

    s = float(riesz_exponent)
    pair_scale = s * np.power(dist2, -0.5 * (s + 2.0))
    pair_scale[~np.isfinite(pair_scale)] = 0.0
    forces = np.sum(delta * pair_scale[:, :, None], axis=1).astype(np.float64)
    force_norm = np.linalg.norm(forces, axis=1)
    return forces, force_norm.astype(np.float64)


def _wrap_periodic(points: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(points, dtype=np.float64), 1.0)


def _reflect_into_unit_cube(points: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(points, dtype=np.float64), 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def _apply_boundary_mode(points: np.ndarray, boundary_mode: str) -> np.ndarray:
    if boundary_mode == "periodic":
        return _wrap_periodic(points)
    if boundary_mode == "reflective":
        return _reflect_into_unit_cube(points)
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")


def _compute_pairwise_displacements(points: np.ndarray, boundary_mode: str) -> np.ndarray:
    if boundary_mode == "periodic":
        return _minimum_image_displacements(points)
    if boundary_mode == "reflective":
        return _bounded_displacements(points)
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")


def compute_riesz_energy(
    points: np.ndarray,
    *,
    boundary_mode: str,
    riesz_exponent: float,
    softening: float,
) -> float:
    if boundary_mode == "periodic":
        return compute_periodic_riesz_energy(
            points,
            riesz_exponent=riesz_exponent,
            softening=softening,
        )
    if boundary_mode == "reflective":
        return compute_bounded_riesz_energy(
            points,
            riesz_exponent=riesz_exponent,
            softening=softening,
        )
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")


def compute_repulsive_forces(
    points: np.ndarray,
    *,
    boundary_mode: str,
    riesz_exponent: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    if boundary_mode == "periodic":
        return compute_periodic_repulsive_forces(
            points,
            riesz_exponent=riesz_exponent,
            softening=softening,
        )
    if boundary_mode == "reflective":
        return compute_bounded_repulsive_forces(
            points,
            riesz_exponent=riesz_exponent,
            softening=softening,
        )
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")


def compute_design_metrics(points: np.ndarray, *, boundary_mode: str = "periodic") -> dict[str, float]:
    delta = _compute_pairwise_displacements(points, boundary_mode)
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return {
        "min_pair_distance": float(np.min(nearest)),
        "mean_nearest_neighbor_distance": float(np.mean(nearest)),
        "max_nearest_neighbor_distance": float(np.max(nearest)),
    }


def _relative_change(new_value: float, old_value: float) -> float:
    denominator = max(abs(float(old_value)), 1.0e-30)
    return float(abs(float(new_value) - float(old_value)) / denominator)


def relax_glass_points(
    config: GlassDesignConfig,
    initial_points: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    points = (
        _generate_initial_unit_points(config)
        if initial_points is None
        else np.asarray(initial_points, dtype=np.float64)
    )
    if points.shape != (int(config.n_points), int(config.theta_dim)):
        raise ValueError(
            "initial_points must have shape "
            f"({int(config.n_points)}, {int(config.theta_dim)}), got {points.shape}."
        )
    points = _apply_boundary_mode(points, config.boundary_mode)
    riesz_exponent = _resolve_riesz_exponent(config)
    step_size = float(config.initial_step_size)
    history: list[dict[str, float]] = []

    current_energy = compute_riesz_energy(
        points,
        boundary_mode=str(config.boundary_mode),
        riesz_exponent=riesz_exponent,
        softening=float(config.softening),
    )
    current_metrics = compute_design_metrics(points, boundary_mode=str(config.boundary_mode))
    stable_counter = 0
    accepted_steps = 0
    rejected_steps = 0
    last_relative_improvement = float("inf")
    last_metric_relative_change = float("inf")
    iterations_completed = 0
    convergence_reason = "max_iterations_reached"
    min_iterations_reached = False

    initial_metrics = dict(current_metrics)
    history.append(
        {
            "iteration": 0.0,
            "energy": float(current_energy),
            "step_size": float(step_size),
            "accepted": 1.0,
            "backtracking_steps": 0.0,
            "max_force_norm": 0.0,
            "mean_force_norm": 0.0,
            **initial_metrics,
        }
    )

    for iteration in range(1, int(config.max_iterations) + 1):
        iterations_completed = int(iteration)
        forces, force_norm = compute_repulsive_forces(
            points,
            boundary_mode=str(config.boundary_mode),
            riesz_exponent=riesz_exponent,
            softening=float(config.softening),
        )
        max_force = float(np.max(force_norm))
        mean_force = float(np.mean(force_norm))
        if not np.isfinite(max_force) or max_force <= 0.0:
            convergence_reason = "nonfinite_or_zero_force"
            break

        normalized_direction = forces / max_force
        accepted = False
        candidate = points
        candidate_energy = float(current_energy)
        trial_step = float(step_size)
        backtracking_steps = 0
        for backtracking_steps in range(int(config.max_backtracking_steps)):
            displacement = trial_step * normalized_direction
            trial_candidate = _apply_boundary_mode(points + displacement, str(config.boundary_mode))
            trial_energy = compute_riesz_energy(
                trial_candidate,
                boundary_mode=str(config.boundary_mode),
                riesz_exponent=riesz_exponent,
                softening=float(config.softening),
            )
            if float(trial_energy) < float(current_energy):
                candidate = trial_candidate
                candidate_energy = float(trial_energy)
                accepted = True
                break
            trial_step *= 0.5
            if trial_step < float(config.min_step_size):
                break

        if accepted:
            prev_energy = float(current_energy)
            prev_metrics = dict(current_metrics)
            points = candidate
            current_energy = float(candidate_energy)
            current_metrics = compute_design_metrics(points, boundary_mode=str(config.boundary_mode))
            accepted_steps += 1
            step_size = min(float(config.max_step_size), float(trial_step * config.step_growth))
            denominator = max(abs(prev_energy), 1.0e-30)
            last_relative_improvement = float((prev_energy - current_energy) / denominator)
            last_metric_relative_change = _relative_change(
                current_metrics["mean_nearest_neighbor_distance"],
                prev_metrics["mean_nearest_neighbor_distance"],
            )
            if (
                int(iteration) >= int(config.min_iterations)
                and last_relative_improvement < float(config.stable_rel_energy_tol)
                and last_metric_relative_change < float(config.stable_rel_metric_tol)
            ):
                stable_counter += 1
            else:
                stable_counter = 0
        else:
            rejected_steps += 1
            step_size = float(max(trial_step, float(config.min_step_size)))
            last_relative_improvement = 0.0
            last_metric_relative_change = 0.0
            stable_counter = 0

        if iteration % int(config.history_stride) == 0 or accepted:
            history.append(
                {
                    "iteration": float(iteration),
                    "energy": float(current_energy),
                    "step_size": float(step_size),
                    "accepted": 1.0 if accepted else 0.0,
                    "backtracking_steps": float(backtracking_steps),
                    "max_force_norm": float(max_force),
                    "mean_force_norm": float(mean_force),
                    **current_metrics,
                }
            )

        min_iterations_reached = int(iteration) >= int(config.min_iterations)
        if step_size <= float(config.min_step_size) and min_iterations_reached:
            convergence_reason = "step_size_below_minimum"
            break
        if (
            min_iterations_reached
            and stable_counter >= int(config.stable_iteration_window)
        ):
            convergence_reason = "stabilized_energy_and_metric"
            break

    final_metrics = dict(current_metrics)
    metadata = {
        "n_points": int(config.n_points),
        "theta_dim": int(config.theta_dim),
        "init_mode": str(config.init_mode),
        "boundary_mode": str(config.boundary_mode),
        "force_model": str(config.force_model),
        "riesz_exponent": float(riesz_exponent),
        "force_decay_exponent": float(riesz_exponent + 1.0),
        "softening": float(config.softening),
        "max_iterations": int(config.max_iterations),
        "min_iterations": int(config.min_iterations),
        "iterations_completed": int(iterations_completed),
        "min_iterations_reached": bool(min_iterations_reached),
        "convergence_reason": str(convergence_reason),
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "final_energy": float(current_energy),
        "last_relative_improvement": float(last_relative_improvement),
        "last_metric_relative_change": float(last_metric_relative_change),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
    }
    return points.astype(np.float64), history, metadata


def generate_glass_design(
    *,
    runtime_config: ValidationRuntimeConfig | None = None,
    glass_config: GlassDesignConfig | None = None,
) -> GlassDesignResult:
    config = GlassDesignConfig() if glass_config is None else glass_config
    runtime = build_default_config() if runtime_config is None else runtime_config
    theta_bounds = theta_bounds_as_array(runtime.theta_bounds)
    if theta_bounds.shape[0] != int(config.theta_dim):
        raise ValueError(
            "glass theta_dim must match runtime theta bounds dimension, "
            f"got {int(config.theta_dim)} vs {theta_bounds.shape[0]}."
        )

    unit_points, history, metadata = relax_glass_points(config)
    raw_points = denormalize_theta_batch(unit_points, theta_bounds)
    metadata = {
        **metadata,
        "theta_bounds": theta_bounds.astype(np.float64).tolist(),
        "theta_names": list(runtime.theta_bounds.keys()),
    }
    return GlassDesignResult(
        unit_points=unit_points.astype(np.float64),
        raw_points=raw_points.astype(np.float64),
        history=history,
        metadata=metadata,
    )


def save_glass_design(result: GlassDesignResult, output_path: str | Path) -> tuple[Path, Path]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".json")
    np.savez_compressed(
        output,
        unit_points=np.asarray(result.unit_points, dtype=np.float64),
        raw_points=np.asarray(result.raw_points, dtype=np.float64),
        history=np.asarray(
            [[entry[key] for key in sorted(entry.keys())] for entry in result.history],
            dtype=np.float64,
        ),
        history_keys=np.asarray(sorted(result.history[0].keys()), dtype=object),
        metadata_json=json.dumps(result.metadata, ensure_ascii=True),
    )
    summary_path.write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "history": result.history,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an 8D glass design by relaxing repulsive particles on the "
            "unit hypercube with periodic boundaries."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config for theta bounds.")
    parser.add_argument("--n-points", type=int, default=128, help="Number of design points.")
    parser.add_argument("--theta-dim", type=int, default=8, help="Parameter-space dimension.")
    parser.add_argument(
        "--init-mode",
        type=str,
        default="sobol",
        choices=("sobol", "random"),
        help="Initial point generator before glass relaxation.",
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default="periodic",
        choices=("periodic", "reflective"),
        help="Boundary handling: flat torus periodic box or reflective hard-wall cube.",
    )
    parser.add_argument(
        "--force-model",
        type=str,
        default="dimension_aware_coulomb",
        choices=("dimension_aware_coulomb", "inverse_square", "riesz"),
        help="Repulsive force model.",
    )
    parser.add_argument(
        "--riesz-exponent",
        type=float,
        default=None,
        help="Custom Riesz energy exponent when --force-model=riesz.",
    )
    parser.add_argument("--seed", type=int, default=20260407, help="Random seed.")
    parser.add_argument("--softening", type=float, default=1.0e-3, help="Pairwise softening radius.")
    parser.add_argument("--max-iterations", type=int, default=4000, help="Maximum relaxation iterations.")
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=512,
        help="Minimum accepted relaxation sweeps before convergence can trigger.",
    )
    parser.add_argument("--initial-step-size", type=float, default=0.05, help="Initial max displacement.")
    parser.add_argument("--max-step-size", type=float, default=0.10, help="Largest allowed max displacement.")
    parser.add_argument("--min-step-size", type=float, default=1.0e-6, help="Stopping threshold for step size.")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/designs/glass_design_8d_128.npz",
        help="Output .npz path. A sibling .json summary is written automatically.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from z2quijote.runtime_core.config import load_config

    runtime_config = (
        build_default_config()
        if args.config is None
        else load_config(Path(args.config).expanduser().resolve())
    )
    glass_config = GlassDesignConfig(
        n_points=int(args.n_points),
        theta_dim=int(args.theta_dim),
        init_mode=str(args.init_mode),
        random_seed=int(args.seed),
        boundary_mode=str(args.boundary_mode),
        force_model=str(args.force_model),
        riesz_exponent=args.riesz_exponent,
        softening=float(args.softening),
        max_iterations=int(args.max_iterations),
        min_iterations=int(args.min_iterations),
        initial_step_size=float(args.initial_step_size),
        max_step_size=float(args.max_step_size),
        min_step_size=float(args.min_step_size),
        output_path=str(args.output),
    )
    result = generate_glass_design(runtime_config=runtime_config, glass_config=glass_config)
    output_path, summary_path = save_glass_design(result, glass_config.output_path or args.output)

    print(f"[glass] saved points to: {output_path}")
    print(f"[glass] saved summary to: {summary_path}")
    print(
        "[glass] final min / mean nearest-neighbor distance: "
        f"{result.metadata['final_metrics']['min_pair_distance']:.6f} / "
        f"{result.metadata['final_metrics']['mean_nearest_neighbor_distance']:.6f}"
    )
    print(
        "[glass] force model / exponents: "
        f"{result.metadata['force_model']} | "
        f"Riesz s={result.metadata['riesz_exponent']:.3f}, "
        f"force decay={result.metadata['force_decay_exponent']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
