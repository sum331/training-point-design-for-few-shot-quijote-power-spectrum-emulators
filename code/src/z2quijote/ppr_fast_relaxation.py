from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np


def _nearest_neighbor_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def _metric_block(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _write_plot(out_dir: Path, history: list[float], steps: list[float], nearest: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    path = out_dir / "lofi_relaxation_diagnostics.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    axes[0].plot(history, linewidth=1.2)
    axes[0].set_title("mean potential")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("U")
    axes[1].plot(steps, linewidth=1.2)
    axes[1].set_title("max displacement")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    axes[2].hist(nearest, bins=24)
    axes[2].set_title("nearest-neighbor distance")
    axes[2].set_xlabel("unit-space distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _torch_device(requested: str):
    import torch

    normalized = str(requested or "auto").strip().lower()
    if normalized in {"auto", "cuda_if_available", "cuda-if-available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PPR relaxation, but torch.cuda is unavailable.")
        return torch.device("cuda")
    return torch.device(normalized)


def _torch_dtype(name: str):
    import torch

    normalized = str(name or "float32").strip().lower()
    if normalized in {"float64", "double"}:
        return torch.float64
    if normalized in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype for PPR relaxation: {name!r}")


class _TorchBiasPotential:
    def __init__(
        self,
        potential: Any,
        *,
        device: Any,
        dtype: Any,
        effective_neighbors: int | None = None,
    ) -> None:
        import torch

        model = getattr(potential, "variance_model", None)
        required = ("theta_unit", "bias", "accepted_count")
        if model is None or any(not hasattr(model, name) for name in required):
            raise TypeError("Fast z2 PPR relaxation requires a BiasFieldScalarModel-like potential.")
        theta = np.asarray(model.theta_unit, dtype=np.float64)
        bias = np.asarray(model.bias, dtype=np.float64).reshape(-1)
        count = np.asarray(model.accepted_count, dtype=np.int64).reshape(-1)
        if theta.ndim != 2 or theta.shape[0] != bias.size or bias.shape != count.shape:
            raise ValueError("Bias support arrays are not aligned.")

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.theta = torch.as_tensor(theta, device=device, dtype=dtype)
        self.bias = torch.as_tensor(np.maximum(bias, 0.0), device=device, dtype=dtype)
        self.count = torch.as_tensor(count, device=device, dtype=dtype)
        self.dim = int(theta.shape[1])
        primary = int(getattr(model, "neighbors", 96))
        fallback = int(getattr(model, "fallback_neighbors", max(primary, 160)))
        k = int(effective_neighbors or max(primary, fallback))
        self.neighbors = max(1, min(k, int(theta.shape[0])))
        self.high_confidence_count = float(max(1, int(getattr(model, "high_confidence_count", 20))))
        self.global_fallback = float(np.nanmedian(bias))
        self.variance_floor = float(getattr(potential, "variance_floor", 1.0e-12))
        self.variance_lo = float(getattr(potential, "_variance_lo", 0.0))
        self.variance_scale = float(max(getattr(potential, "_variance_scale", 1.0), self.variance_floor))
        refs = np.asarray(getattr(potential, "_rank_reference_values", np.empty(0)), dtype=np.float64)
        refs = refs[np.isfinite(refs)]
        self.rank_refs = torch.as_tensor(np.sort(refs), device=device, dtype=dtype)
        self.rank_body_enabled = bool(getattr(potential, "rank_body_enabled", False))
        self.rank_body_quantile = float(getattr(potential, "rank_body_quantile", 0.68))
        self.rank_body_width = float(getattr(potential, "rank_body_width", 0.36))
        self.rank_body_floor = float(getattr(potential, "rank_body_floor", 0.25))
        self.rank_body_mix = float(getattr(potential, "rank_body_mix", 1.0))
        self.variance_gain = float(getattr(potential, "variance_gain", 1.0))
        self.variance_power = float(getattr(potential, "variance_power", 1.0))
        soft_cap = getattr(potential, "variance_soft_cap", None)
        self.variance_soft_cap = None if soft_cap is None else float(soft_cap)

    def input_length_scales(self) -> np.ndarray:
        return np.ones((self.dim,), dtype=np.float64)

    def raw(self, x) -> Any:
        torch = self.torch
        query = torch.clamp(x, 0.0, 1.0)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        diff = query[:, None, :] - self.theta[None, :, :]
        dist2 = torch.sum(diff * diff, dim=2)
        values2, indices = torch.topk(dist2, k=self.neighbors, dim=1, largest=False, sorted=True)
        distances = torch.sqrt(torch.clamp(values2, min=0.0))
        bandwidth = torch.clamp(distances[:, [-1]], min=torch.finfo(self.dtype).eps)
        reliability = torch.clamp(self.count[indices] / self.high_confidence_count, max=1.0)
        weights = reliability * torch.exp(-0.5 * (distances / bandwidth).square())
        total = torch.sum(weights, dim=1)
        support_values = self.bias[indices]
        fallback = torch.full_like(total, float(self.global_fallback))
        out = torch.where(
            total > 0.0,
            torch.sum(weights * support_values, dim=1) / torch.clamp(total, min=torch.finfo(self.dtype).eps),
            fallback,
        )
        exact = distances[:, 0] <= 1.0e-7
        if bool(torch.any(exact)):
            out = torch.where(exact, support_values[:, 0], out)
        return torch.clamp(out, min=0.0)

    def raw_normalized(self, x) -> Any:
        values = self.raw(x)
        values = self.torch.clamp(values, min=self.variance_floor)
        return self.torch.clamp((values - self.variance_lo) / self.variance_scale, min=0.0)

    def _rank_multiplier(self, raw_values) -> Any:
        torch = self.torch
        if not self.rank_body_enabled or self.rank_refs.numel() == 0:
            return torch.ones_like(raw_values)
        refs = self.rank_refs
        n = int(refs.numel())
        if n <= 1:
            return torch.ones_like(raw_values)
        clipped = torch.clamp(raw_values.reshape(-1), min=self.variance_floor)
        pos = torch.searchsorted(refs, clipped, right=False)
        pos = torch.clamp(pos, 1, n - 1)
        left = refs[pos - 1]
        right = refs[pos]
        t = (clipped - left) / torch.clamp(right - left, min=torch.finfo(self.dtype).eps)
        left_rank = (pos.to(self.dtype) - 0.5) / float(n)
        right_rank = (pos.to(self.dtype) + 0.5) / float(n)
        ranks = left_rank + t * (right_rank - left_rank)
        quantile = min(1.0, max(0.0, self.rank_body_quantile))
        half_width = 0.5 * min(1.0, max(1.0e-6, self.rank_body_width))
        scaled = torch.abs(ranks - quantile) / max(half_width, 1.0e-6)
        window = torch.where(
            scaled <= 1.0,
            0.5 * (1.0 + torch.cos(torch.pi * scaled)),
            torch.zeros_like(scaled),
        )
        floor = min(1.0, max(0.0, self.rank_body_floor))
        mix = min(1.0, max(0.0, self.rank_body_mix))
        body = floor + (1.0 - floor) * window
        return ((1.0 - mix) + mix * body).reshape(raw_values.shape)

    def mapped_variance(self, x) -> Any:
        torch = self.torch
        raw = torch.clamp(self.raw(x), min=self.variance_floor)
        q = torch.clamp((raw - self.variance_lo) / self.variance_scale, min=0.0)
        if self.variance_power != 1.0:
            q = torch.pow(torch.clamp(q, min=1.0e-30), self.variance_power)
        if self.variance_soft_cap is not None and self.variance_soft_cap > 0.0:
            cap = float(self.variance_soft_cap)
            q = cap * q / (cap + q)
        q = q * self._rank_multiplier(raw)
        return float(self.variance_gain) * q

    def finite_difference_force(self, x, eps: float) -> Any:
        torch = self.torch
        n, dim = int(x.shape[0]), int(x.shape[1])
        h = max(float(eps), 1.0e-5)
        points = []
        denominators = []
        for col in range(dim):
            xp = x.clone()
            xm = x.clone()
            xp[:, col] = torch.clamp(xp[:, col] + h, 0.0, 1.0)
            xm[:, col] = torch.clamp(xm[:, col] - h, 0.0, 1.0)
            points.extend([xp, xm])
            denominators.append(torch.clamp(xp[:, col] - xm[:, col], min=1.0e-12))
        values = self.mapped_variance(torch.cat(points, dim=0)).reshape(2 * dim, n)
        grad = torch.zeros_like(x)
        for col in range(dim):
            grad[:, col] = (values[2 * col] - values[2 * col + 1]) / denominators[col]
        return grad


def _cap_norm_torch(force, cap: float):
    torch = __import__("torch")
    cap = float(cap)
    if cap <= 0.0:
        return force
    norm = torch.linalg.norm(force, dim=1, keepdim=True)
    scale = torch.clamp(cap / torch.clamp(norm, min=1.0e-12), max=1.0)
    return force * scale


def _metric_scales(settings: Any, dim: int, potential_runtime: _TorchBiasPotential):
    torch = potential_runtime.torch
    cfg = settings.repulsion
    metric = str(getattr(cfg, "metric", "euclidean")).strip().lower()
    if metric in {"euclidean", "unit", ""}:
        scales = np.ones(int(dim), dtype=np.float64)
    elif metric in {"configured", "manual"}:
        if cfg.metric_length_scales is None:
            raise ValueError("repulsion.metric_length_scales is required for configured metric.")
        scales = np.asarray(cfg.metric_length_scales, dtype=np.float64).reshape(-1)
    elif metric in {"gp_lengthscale", "potential_lengthscale", "lengthscale"}:
        scales = potential_runtime.input_length_scales()
    else:
        raise ValueError(f"unsupported repulsion metric: {cfg.metric!r}")
    if scales.size != int(dim):
        raise ValueError(f"repulsion metric expected {dim} length scales, got {scales.size}.")
    geom = float(np.exp(np.mean(np.log(scales))))
    scales = scales / max(geom, 1.0e-12)
    scales = np.clip(
        scales,
        float(getattr(cfg, "metric_scale_floor", 0.05)),
        float(getattr(cfg, "metric_scale_ceiling", 20.0)),
    )
    return torch.as_tensor(scales, device=potential_runtime.device, dtype=potential_runtime.dtype)


def _repulsion_force(x, settings: Any, potential_runtime: _TorchBiasPotential, metric_scales):
    torch = potential_runtime.torch
    cfg = settings.repulsion
    if not bool(cfg.enabled) or float(cfg.strength) <= 0.0:
        return torch.zeros_like(x)
    diff = x[:, None, :] - x[None, :, :]
    scaled = diff / metric_scales[None, None, :]
    dist2 = torch.sum(scaled * scaled, dim=2)
    eye = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
    dist2 = torch.where(eye, torch.full_like(dist2, float("inf")), dist2)
    if int(cfg.nearest_neighbors) > 0 and int(cfg.nearest_neighbors) < x.shape[0] - 1:
        kth = int(cfg.nearest_neighbors)
        cutoff = torch.kthvalue(dist2, kth + 1, dim=1).values.reshape(-1, 1)
        mask = dist2 <= cutoff
    else:
        mask = torch.isfinite(dist2)
    length2 = float(cfg.length_scale) ** 2
    soft2 = float(cfg.softening) ** 2
    weights = torch.exp(-0.5 * dist2 / max(length2, 1.0e-12)) / torch.clamp(dist2 + soft2, min=1.0e-12)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    direction = diff / torch.square(metric_scales)[None, None, :]
    force = float(cfg.strength) * torch.sum(weights[:, :, None] * direction, dim=1)
    if bool(getattr(cfg, "adaptive_to_potential", False)):
        q = torch.clamp(potential_runtime.raw_normalized(x), 0.0, 1.0)
        power = max(float(getattr(cfg, "adaptive_power", 1.0)), 0.0)
        minimum = min(1.0, max(0.0, float(getattr(cfg, "adaptive_min_multiplier", 0.25))))
        multiplier = torch.clamp(torch.pow(1.0 - q, power), min=minimum).reshape(-1, 1)
        force = force * multiplier
    return _cap_norm_torch(force, float(cfg.force_cap))


def _boundary_force(x, settings: Any):
    torch = __import__("torch")
    cfg = settings.boundary
    margin = float(cfg.margin)
    if margin <= 0.0 or float(cfg.strength) <= 0.0:
        return torch.zeros_like(x)
    low = x < margin
    high = x > 1.0 - margin
    force = torch.zeros_like(x)
    force = force + low * ((margin - x) / margin)
    force = force - high * ((x - (1.0 - margin)) / margin)
    return _cap_norm_torch(float(cfg.strength) * force, float(cfg.force_cap))


def _clip_to_boundary_guard(x, settings: Any):
    torch = __import__("torch")
    cfg = settings.boundary
    margin = float(cfg.margin)
    fraction = float(cfg.hard_clip_margin_fraction)
    if margin <= 0.0 or fraction <= 0.0:
        return torch.clamp(x, 0.0, 1.0)
    guard = min(max(margin * fraction, 0.0), 0.49)
    return torch.clamp(x, guard, 1.0 - guard)


def run_fast_lofi_relaxation_from_bias_potential(
    config: Any,
    potential_path: str | Path,
    out_dir: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
    effective_neighbors: int | None = None,
) -> dict[str, Any]:
    import torch

    from r2_multi_al.parameter_space import denormalize_theta
    from r2_multi_al.pipeline import (
        _sample_unit,
        make_bounds,
        make_configured_k_grid,
        make_rng,
        potential_mapping_settings_from_config,
        potential_scalar_weights_from_config,
        potential_weighting_settings_from_config,
        relaxation_settings_from_config,
    )

    potential_file = Path(potential_path).resolve()
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with potential_file.open("rb") as handle:
        potential = pickle.load(handle)
    for key, value in potential_mapping_settings_from_config(config).items():
        setattr(potential, key, value)
    k_path = potential_file.parent / "k_bins.npy"
    k_bins = np.load(k_path).astype(np.float64) if k_path.exists() else make_configured_k_grid(config)
    if hasattr(potential, "set_scalar_variance_weights"):
        potential.set_scalar_variance_weights(
            potential_scalar_weights_from_config(config, k_bins),
            recalibrate=True,
        )
    runtime = _TorchBiasPotential(
        potential,
        device=_torch_device(device),
        dtype=_torch_dtype(dtype),
        effective_neighbors=effective_neighbors,
    )
    rng = make_rng(config)
    bounds = make_bounds(config)
    settings = relaxation_settings_from_config(config)
    initial = _sample_unit(
        int(settings.particles),
        bounds.shape[0],
        method=str(settings.initial_sampling),
        rng=rng,
    )
    torch_device = runtime.device
    torch_dtype = runtime.dtype
    with torch.no_grad():
        x0 = torch.as_tensor(initial, device=torch_device, dtype=torch_dtype)
        x = _clip_to_boundary_guard(x0.clone(), settings)
        if float(settings.jitter) > 0.0:
            jitter = torch.as_tensor(
                rng.normal(0.0, float(settings.jitter), size=initial.shape),
                device=torch_device,
                dtype=torch_dtype,
            )
            x = _clip_to_boundary_guard(x + jitter, settings)
        v = torch.zeros_like(x)
        metric_scales = _metric_scales(settings, x.shape[1], runtime)
        potential_history: list[float] = []
        max_step_history: list[float] = []
        converged = False
        for step in range(int(settings.max_steps)):
            p_force = runtime.finite_difference_force(x, eps=float(settings.finite_difference_eps))
            p_force = _cap_norm_torch(p_force, float(settings.potential_force_cap))
            force = p_force + _repulsion_force(x, settings, runtime, metric_scales) + _boundary_force(x, settings)
            v = float(settings.damping) * v + float(settings.dt) * force
            dx = float(settings.dt) * v
            raw_new = x + dx
            x_new = _clip_to_boundary_guard(raw_new, settings)
            hit = torch.abs(x_new - raw_new) > 0.0
            v = torch.where(hit, torch.zeros_like(v), v)
            max_step = float(torch.max(torch.linalg.norm(x_new - x, dim=1)).detach().cpu().item())
            x = x_new
            potential_history.append(float((-torch.mean(runtime.mapped_variance(x))).detach().cpu().item()))
            max_step_history.append(max_step)
            if step + 1 >= int(settings.min_steps) and max_step < float(settings.convergence_tol):
                converged = True
                break
        final_unit = x.detach().cpu().numpy().astype(np.float64)
        initial_unit = initial.astype(np.float64)
        final_variance = runtime.mapped_variance(x).detach().cpu().numpy().astype(np.float64)
        initial_variance = runtime.mapped_variance(x0).detach().cpu().numpy().astype(np.float64)
        raw_final_variance = runtime.raw_normalized(x).detach().cpu().numpy().astype(np.float64)
        raw_initial_variance = runtime.raw_normalized(x0).detach().cpu().numpy().astype(np.float64)
    nearest = _nearest_neighbor_distances(final_unit)
    np.savez_compressed(
        output / "lofi_design.npz",
        theta_unit_initial=initial_unit,
        theta_unit=final_unit,
        theta_raw=denormalize_theta(final_unit, bounds),
        potential_history=np.asarray(potential_history, dtype=np.float64),
        max_step_history=np.asarray(max_step_history, dtype=np.float64),
        nearest_neighbor_distance=nearest.astype(np.float64),
        final_normalized_variance=np.asarray(final_variance, dtype=np.float64),
        initial_normalized_variance=np.asarray(initial_variance, dtype=np.float64),
        final_raw_normalized_variance=np.asarray(raw_final_variance, dtype=np.float64),
        initial_raw_normalized_variance=np.asarray(raw_initial_variance, dtype=np.float64),
    )
    plot_path = _write_plot(output, potential_history, max_step_history, nearest)
    boundary_margin = float(settings.boundary.margin)
    elapsed = time.time() - started
    summary = {
        "status": "ok",
        "backend": "z2_torch_knn_bias_relaxation",
        "potential_path": str(potential_file),
        "converged": bool(converged),
        "steps": int(len(max_step_history)),
        "particles": int(final_unit.shape[0]),
        "theta_dim": int(final_unit.shape[1]),
        "device": str(torch_device),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "support_points": int(runtime.theta.shape[0]),
        "effective_neighbors": int(runtime.neighbors),
        "elapsed_seconds": float(elapsed),
        "settings": json.loads(json.dumps(asdict(settings))),
        "potential_mapping": potential_mapping_settings_from_config(config),
        "potential_weighting": potential_weighting_settings_from_config(config),
        "final_min_coordinate": float(np.min(final_unit)),
        "final_max_coordinate": float(np.max(final_unit)),
        "boundary_fraction_inside_margin": float(
            np.mean((final_unit < boundary_margin) | (final_unit > 1.0 - boundary_margin))
        ),
        "nearest_neighbor_distance": _metric_block(nearest),
        "initial_normalized_variance": _metric_block(initial_variance),
        "final_normalized_variance": _metric_block(final_variance),
        "initial_raw_normalized_variance": _metric_block(raw_initial_variance),
        "final_raw_normalized_variance": _metric_block(raw_final_variance),
        "mean_potential_initial": float(potential_history[0]),
        "mean_potential_final": float(potential_history[-1]),
        "last_max_step": float(max_step_history[-1]),
        "plot_path": str(plot_path),
    }
    (output / "lofi_relaxation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
