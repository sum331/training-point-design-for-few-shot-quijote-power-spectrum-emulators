from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from .config import ModelConfig
from .parameter_space import ParameterSpace


@dataclass(slots=True)
class Prediction:
    theta_raw: np.ndarray
    k_bins: np.ndarray
    log_pk_mean: np.ndarray
    pk_mean: np.ndarray
    pc_mean: np.ndarray
    pc_std: np.ndarray


@dataclass(slots=True)
class PCAGPDirectCDMEmulator:
    parameter_space: ParameterSpace
    model_config: ModelConfig
    target_kind: str = "direct_cdm_logpk"
    pca: PCA | None = None
    gp_models: list[GaussianProcessRegressor] | None = None
    score_mean: np.ndarray | None = None
    score_std: np.ndarray | None = None
    k_bins: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def fit(self, theta_raw: np.ndarray, log_pk: np.ndarray, k_bins: np.ndarray) -> "PCAGPDirectCDMEmulator":
        theta = np.asarray(theta_raw, dtype=np.float64)
        y = np.asarray(log_pk, dtype=np.float64)
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if theta.ndim != 2 or theta.shape[1] != self.parameter_space.dim:
            raise ValueError(f"theta_raw must have shape [N,{self.parameter_space.dim}], got {theta.shape}.")
        if y.ndim != 2 or y.shape[0] != theta.shape[0] or y.shape[1] != k.shape[0]:
            raise ValueError(f"log_pk must have shape [{theta.shape[0]},{k.shape[0]}], got {y.shape}.")
        if theta.shape[0] < 2:
            raise ValueError("PCAGPDirectCDMEmulator requires at least two training points.")
        unit = self.parameter_space.normalize(theta, clip=False)
        n_components = int(min(max(1, self.model_config.pca_components), y.shape[0], y.shape[1]))
        pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
        scores = np.asarray(pca.fit_transform(y), dtype=np.float64)
        score_mean = np.mean(scores, axis=0)
        score_std = np.maximum(np.std(scores, axis=0), 1.0e-12)
        scaled = (scores - score_mean.reshape(1, -1)) / score_std.reshape(1, -1)
        gps: list[GaussianProcessRegressor] = []
        for index in range(n_components):
            gp = GaussianProcessRegressor(
                kernel=self._kernel(theta_dim=unit.shape[1]),
                alpha=float(self.model_config.gp_alpha),
                normalize_y=bool(self.model_config.normalize_y),
                random_state=1000 + index,
                n_restarts_optimizer=int(self.model_config.gp_n_restarts_optimizer),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(unit, scaled[:, index])
            gps.append(gp)
        self.pca = pca
        self.gp_models = gps
        self.score_mean = score_mean.astype(np.float64)
        self.score_std = score_std.astype(np.float64)
        self.k_bins = k.astype(np.float64)
        self.metadata = {
            "emulator_kind": f"pca_gp_{str(self.target_kind)}",
            "target_kind": str(self.target_kind),
            "training_points": int(theta.shape[0]),
            "k_count": int(k.shape[0]),
            "pca_components": int(n_components),
            "gp_alpha": float(self.model_config.gp_alpha),
        }
        return self

    def predict(self, theta_raw: np.ndarray) -> Prediction:
        self._check_fitted()
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        unit = self.parameter_space.normalize(theta, clip=False)
        pc_mean, pc_std = self.predict_pc_stats(theta)
        assert self.pca is not None
        assert self.k_bins is not None
        log_pk = np.asarray(self.pca.inverse_transform(pc_mean), dtype=np.float64)
        pk = np.exp(log_pk)
        return Prediction(
            theta_raw=theta,
            k_bins=self.k_bins,
            log_pk_mean=log_pk,
            pk_mean=pk,
            pc_mean=pc_mean,
            pc_std=pc_std,
        )

    def predict_pc_stats(self, theta_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        unit = self.parameter_space.normalize(theta)
        assert self.gp_models is not None
        assert self.score_mean is not None
        assert self.score_std is not None
        mean_cols: list[np.ndarray] = []
        std_cols: list[np.ndarray] = []
        for gp in self.gp_models:
            mean, std = gp.predict(unit, return_std=True)
            mean_cols.append(np.asarray(mean, dtype=np.float64).reshape(-1, 1))
            std_cols.append(np.asarray(std, dtype=np.float64).reshape(-1, 1))
        scaled_mean = np.hstack(mean_cols)
        scaled_std = np.hstack(std_cols)
        pc_mean = scaled_mean * self.score_std.reshape(1, -1) + self.score_mean.reshape(1, -1)
        pc_std = np.maximum(scaled_std, 0.0) * self.score_std.reshape(1, -1)
        return pc_mean.astype(np.float64), pc_std.astype(np.float64)

    def uncertainty_scalar(self, theta_raw: np.ndarray) -> np.ndarray:
        _, pc_std = self.predict_pc_stats(theta_raw)
        return np.sqrt(np.mean(pc_std**2, axis=1))

    def _kernel(self, *, theta_dim: int):
        return ConstantKernel(
            constant_value=float(self.model_config.constant_value),
            constant_value_bounds=self.model_config.constant_value_bounds,
        ) * RBF(
            length_scale=np.full((int(theta_dim),), float(self.model_config.length_scale_initial)),
            length_scale_bounds=self.model_config.length_scale_bounds,
        )

    def _check_fitted(self) -> None:
        if self.pca is None or self.gp_models is None or self.k_bins is None:
            raise RuntimeError("PCAGPDirectCDMEmulator is not fitted.")
