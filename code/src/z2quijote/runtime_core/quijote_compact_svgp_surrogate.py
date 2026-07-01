"""Compact full-bank SVGP truth-generator helpers for Quijote BSQ spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
import torch
import gpytorch

from z2quijote.runtime_core.quijote_gp_surrogate import (
    QUIJOTE_BSQ5_PARAMETER_SPACE,
    QUIJOTE_BSQ_THETA_NAMES,
    QuijoteBank,
    _install_legacy_pickle_aliases,
    _interp_logk_batch,
    denormalize_quijote_theta_batch,
    derive_theta_bounds_from_bank,
    ensure_quijote_theta_batch,
    normalize_quijote_theta_batch,
    quijote_theta_bounds_as_array,
)

FloatArray = np.ndarray


class _BatchedSVGPModel(gpytorch.models.ApproximateGP):
    """Independent batched SVGPs that share inputs and model PCA score columns."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        if inducing_points.ndim != 3:
            raise ValueError(
                "inducing_points must have shape [num_outputs, num_inducing, theta_dim]."
            )
        num_outputs = int(inducing_points.shape[0])
        num_inducing = int(inducing_points.shape[1])
        theta_dim = int(inducing_points.shape[2])
        batch_shape = torch.Size([num_outputs])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.num_outputs = num_outputs
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=theta_dim,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.num_outputs, -1, -1)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _resolve_device(raw_device: str | torch.device) -> torch.device:
    text = str(raw_device).strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _torch_state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class QuijoteCompactSVGPSurrogate:
    """Frozen PCA+SVGP nonlinear truth generator trained on Quijote logP."""

    theta_bounds: FloatArray
    theta_names: tuple[str, ...]
    k_bins: FloatArray
    pca_mean: FloatArray
    pca_components: FloatArray
    score_mean: FloatArray
    score_std: FloatArray
    model_state_dict: dict[str, torch.Tensor]
    likelihood_state_dict: dict[str, torch.Tensor]
    num_inducing: int
    power_eps: float
    metadata: dict[str, Any] = field(default_factory=dict)
    _model_cache: Any = field(default=None, init=False, repr=False)
    _likelihood_cache: Any = field(default=None, init=False, repr=False)
    _cache_device: str | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_model_cache"] = None
        state["_likelihood_cache"] = None
        state["_cache_device"] = None
        return state

    @property
    def pca_component_count(self) -> int:
        return int(np.asarray(self.pca_components).shape[0])

    def _initial_inducing_points(self, device: torch.device) -> torch.Tensor:
        count = self.pca_component_count
        theta_dim = int(np.asarray(self.theta_bounds).shape[0])
        return torch.zeros(
            (count, int(self.num_inducing), theta_dim),
            dtype=torch.float32,
            device=device,
        )

    def _runtime_model(
        self,
        device: torch.device,
    ) -> tuple[_BatchedSVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
        cache_key = str(device)
        if (
            self._model_cache is not None
            and self._likelihood_cache is not None
            and self._cache_device == cache_key
        ):
            return self._model_cache, self._likelihood_cache

        model = _BatchedSVGPModel(self._initial_inducing_points(device))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([self.pca_component_count])
        )
        model.load_state_dict(self.model_state_dict)
        likelihood.load_state_dict(self.likelihood_state_dict)
        model = model.to(device)
        likelihood = likelihood.to(device)
        model.eval()
        likelihood.eval()
        self._model_cache = model
        self._likelihood_cache = likelihood
        self._cache_device = cache_key
        return model, likelihood

    def predict(
        self,
        theta_batch: np.ndarray,
        *,
        input_space: str = "raw",
        k_target: np.ndarray | None = None,
        return_std: bool = False,
        device: str | torch.device = "cpu",
    ) -> dict[str, np.ndarray]:
        """Predict nonlinear Quijote P(k) for raw or unit-space 5D theta rows."""

        raw_thetas, unit_thetas = ensure_quijote_theta_batch(
            theta_batch,
            self.theta_bounds,
            input_space=input_space,
        )
        resolved_device = _resolve_device(device)
        model, _ = self._runtime_model(resolved_device)
        x_tensor = torch.as_tensor(unit_thetas, dtype=torch.float32, device=resolved_device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(x_tensor)
            pc_scaled_mean = output.mean.detach().cpu().numpy()
            pc_scaled_var = output.variance.detach().cpu().numpy()
        if pc_scaled_mean.ndim != 2:
            raise RuntimeError(f"Unexpected SVGP mean shape: {pc_scaled_mean.shape}.")
        pc_mean = pc_scaled_mean.T * self.score_std.reshape(1, -1) + self.score_mean.reshape(1, -1)
        pc_std = np.sqrt(np.maximum(pc_scaled_var.T, 0.0)) * self.score_std.reshape(1, -1)
        log_pk_source = (
            pc_mean @ np.asarray(self.pca_components, dtype=np.float64)
            + np.asarray(self.pca_mean, dtype=np.float64).reshape(1, -1)
        )
        source_k = np.asarray(self.k_bins, dtype=np.float64).reshape(-1)
        resolved_k = source_k if k_target is None else np.asarray(k_target, dtype=np.float64).reshape(-1)
        if np.any(resolved_k <= 0.0):
            raise ValueError("k_target must be strictly positive.")
        log_pk = _interp_logk_batch(log_pk_source, source_k, resolved_k)
        pk = np.exp(log_pk)
        result = {
            "raw_thetas": raw_thetas.astype(np.float64),
            "unit_thetas": unit_thetas.astype(np.float64),
            "k_bins": resolved_k.astype(np.float64),
            "log_pk_mean": log_pk.astype(np.float64),
            "pk_mean": pk.astype(np.float64),
            "pc_mean": pc_mean.astype(np.float64),
        }
        if return_std:
            result["pc_std"] = pc_std.astype(np.float64)
        return result


def _select_training_indices(
    point_count: int,
    *,
    train_size: int | None,
    seed: int,
) -> np.ndarray:
    if train_size is None:
        return np.arange(int(point_count), dtype=np.int64)
    resolved_size = int(min(max(1, int(train_size)), int(point_count)))
    rng = np.random.default_rng(int(seed))
    indices = np.asarray(rng.choice(point_count, size=resolved_size, replace=False), dtype=np.int64)
    indices.sort()
    return indices


def _fit_direct_logpk_pca(
    log_pk: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> tuple[PCA, np.ndarray]:
    resolved = int(min(max(1, int(n_components)), log_pk.shape[0], log_pk.shape[1]))
    pca = PCA(n_components=resolved, svd_solver="randomized", random_state=int(seed))
    scores = np.asarray(pca.fit_transform(log_pk), dtype=np.float32)
    return pca, scores


def train_quijote_compact_svgp_surrogate(
    bank: QuijoteBank,
    *,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | None = None,
    train_size: int | None = None,
    train_seed: int = 20260522,
    k_max: float | None = 5.0,
    pca_components: int = 16,
    inducing_points: int = 1024,
    epochs: int = 600,
    batch_size: int = 2048,
    learning_rate: float = 1.0e-2,
    power_eps: float = 1.0e-12,
    device: str | torch.device = "auto",
    progress_every: int = 10,
    progress_callback: Any | None = None,
) -> QuijoteCompactSVGPSurrogate:
    """Train a compact direct-logP PCA+SVGP truth generator from a Quijote bank."""

    torch.manual_seed(int(train_seed))
    np.random.seed(int(train_seed))
    resolved_device = _resolve_device(device)
    raw_thetas = np.asarray(bank.raw_thetas, dtype=np.float64)
    pk_batch = np.asarray(bank.p_nonlin_batch, dtype=np.float32)
    k_bins = np.asarray(bank.k_bins, dtype=np.float64).reshape(-1)
    if k_max is not None:
        k_mask = k_bins <= float(k_max) * (1.0 + 1.0e-12)
        if not np.any(k_mask):
            raise ValueError(f"No Quijote k bins are <= k_max={k_max}.")
    else:
        k_mask = np.ones_like(k_bins, dtype=bool)
    selected_k = k_bins[k_mask].astype(np.float64)
    selected_indices = _select_training_indices(
        raw_thetas.shape[0],
        train_size=train_size,
        seed=int(train_seed),
    )
    train_thetas = raw_thetas[selected_indices]
    train_pk = pk_batch[np.ix_(selected_indices, k_mask)]
    bounds = (
        derive_theta_bounds_from_bank(bank)
        if theta_bounds is None
        else quijote_theta_bounds_as_array(theta_bounds)
    )
    unit_thetas = normalize_quijote_theta_batch(train_thetas, bounds).astype(np.float32)
    log_pk = np.log(np.maximum(train_pk, float(max(power_eps, 1.0e-30)))).astype(np.float32)
    pca_model, raw_scores = _fit_direct_logpk_pca(
        log_pk,
        n_components=int(pca_components),
        seed=int(train_seed),
    )
    score_mean = np.mean(raw_scores, axis=0, dtype=np.float64).astype(np.float32)
    score_std = np.maximum(np.std(raw_scores, axis=0, dtype=np.float64), 1.0e-6).astype(np.float32)
    scaled_scores = ((raw_scores - score_mean.reshape(1, -1)) / score_std.reshape(1, -1)).astype(
        np.float32
    )

    rng = np.random.default_rng(int(train_seed) + 17)
    inducing_count = int(min(max(1, int(inducing_points)), unit_thetas.shape[0]))
    inducing_indices = np.asarray(
        rng.choice(unit_thetas.shape[0], size=inducing_count, replace=False),
        dtype=np.int64,
    )
    base_inducing = torch.as_tensor(
        unit_thetas[inducing_indices],
        dtype=torch.float32,
        device=resolved_device,
    )
    inducing_tensor = base_inducing.unsqueeze(0).expand(scaled_scores.shape[1], -1, -1).contiguous()
    model = _BatchedSVGPModel(inducing_tensor).to(resolved_device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=torch.Size([scaled_scores.shape[1]])
    ).to(resolved_device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=float(learning_rate),
    )
    mll = gpytorch.mlls.VariationalELBO(
        likelihood,
        model,
        num_data=int(unit_thetas.shape[0]),
    )
    x_tensor = torch.as_tensor(unit_thetas, dtype=torch.float32)
    y_tensor = torch.as_tensor(scaled_scores, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(int(train_seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(max(1, batch_size)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    loss_history: list[float] = []
    for epoch in range(1, int(max(1, epochs)) + 1):
        running = 0.0
        batch_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            batch_y = batch_y.to(resolved_device, non_blocking=True).T.contiguous()
            output = model(batch_x)
            loss = -mll(output, batch_y).mean()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batch_count += 1
        mean_loss = running / max(batch_count, 1)
        loss_history.append(mean_loss)
        if progress_callback is not None and (
            epoch == 1
            or epoch == int(max(1, epochs))
            or epoch % int(max(1, progress_every)) == 0
        ):
            progress_callback("quijote_compact_svgp_epoch", epoch, int(max(1, epochs)), mean_loss)

    model.eval()
    likelihood.eval()
    metadata = {
        "surrogate_kind": "quijote_bsq_direct_logpk_pca_svgp",
        "role": "truth_generator",
        "parameter_space": QUIJOTE_BSQ5_PARAMETER_SPACE,
        "theta_names": list(QUIJOTE_BSQ_THETA_NAMES),
        "theta_dim": int(bounds.shape[0]),
        "training_target": "direct_logpk",
        "target_transform": "direct_logpk",
        "linear_anchor_inside_generator": False,
        "used_for_downstream_validation_truth": True,
        "source_bank_size": int(raw_thetas.shape[0]),
        "train_size": int(train_thetas.shape[0]),
        "train_uses_full_bank": bool(train_thetas.shape[0] == raw_thetas.shape[0]),
        "source_k_bin_count": int(k_bins.shape[0]),
        "k_bin_count": int(selected_k.shape[0]),
        "k_min": float(np.min(selected_k)),
        "k_max": float(np.max(selected_k)),
        "k_trust_max": float(k_max) if k_max is not None else float(np.max(selected_k)),
        "pca_components_requested": int(pca_components),
        "pca_components": int(scaled_scores.shape[1]),
        "pca_explained_variance_ratio": np.asarray(
            pca_model.explained_variance_ratio_,
            dtype=np.float64,
        ).tolist(),
        "inducing_points": int(inducing_count),
        "epochs": int(max(1, epochs)),
        "batch_size": int(max(1, batch_size)),
        "learning_rate": float(learning_rate),
        "device": str(resolved_device),
        "train_seed": int(train_seed),
        "power_eps": float(power_eps),
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "loss_history_tail": [float(item) for item in loss_history[-20:]],
        "selected_bank_indices_preview": [int(item) for item in selected_indices[:20].tolist()],
        "bank_metadata": dict(bank.metadata),
    }
    return QuijoteCompactSVGPSurrogate(
        theta_bounds=np.asarray(bounds, dtype=np.float64),
        theta_names=QUIJOTE_BSQ_THETA_NAMES,
        k_bins=selected_k.astype(np.float64),
        pca_mean=np.asarray(pca_model.mean_, dtype=np.float64),
        pca_components=np.asarray(pca_model.components_, dtype=np.float64),
        score_mean=np.asarray(score_mean, dtype=np.float64),
        score_std=np.asarray(score_std, dtype=np.float64),
        model_state_dict=_torch_state_to_cpu(model.state_dict()),
        likelihood_state_dict=_torch_state_to_cpu(likelihood.state_dict()),
        num_inducing=int(inducing_count),
        power_eps=float(power_eps),
        metadata=metadata,
    )


def save_quijote_compact_svgp_surrogate(
    surrogate: QuijoteCompactSVGPSurrogate,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(surrogate, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = dict(surrogate.metadata)
    metadata.update(
        {
            "artifact_path": str(path),
            "theta_bounds": np.asarray(surrogate.theta_bounds, dtype=np.float64).tolist(),
            "k_min": float(np.min(surrogate.k_bins)),
            "k_max": float(np.max(surrogate.k_bins)),
            "pca_component_count": int(surrogate.pca_component_count),
            "num_inducing": int(surrogate.num_inducing),
        }
    )
    path.with_suffix(".json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_quijote_compact_svgp_surrogate(path: str | Path) -> QuijoteCompactSVGPSurrogate:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Quijote compact SVGP surrogate not found: {resolved}")
    _install_legacy_pickle_aliases(
        {"quijote_compact_svgp_surrogate": sys.modules[__name__]}
    )
    with resolved.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, QuijoteCompactSVGPSurrogate):
        raise TypeError(
            f"Expected QuijoteCompactSVGPSurrogate in {resolved}, got {type(loaded).__name__}."
        )
    return loaded


__all__ = [
    "QuijoteCompactSVGPSurrogate",
    "load_quijote_compact_svgp_surrogate",
    "save_quijote_compact_svgp_surrogate",
    "train_quijote_compact_svgp_surrogate",
]
