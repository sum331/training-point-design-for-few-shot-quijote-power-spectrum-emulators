from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import interp1d

from .config import Z2Config
from .csst import data_provider as csst_data_provider
from .direct_cdm import CambCDMAnchorProvider, SpectraBatch
from .emulator import PCAGPDirectCDMEmulator
from .theta_transform import active_to_csst8_theta


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "cublas" in text or "allocation" in text)


class CSSTCDMCAMBAnchorProvider:
    """CAMB/HMCODE2020 CDM nonlinear anchor for CSST 8D theta rows."""

    provider_name = "csst_camb_cdm_hmcode2020"
    power_label = "cdm_hmcode2020"
    matter_power_var = "delta_cdm"

    def __init__(
        self,
        *,
        power_eps: float = 1.0e-12,
        halofit_version: str = "mead2020",
        min_camb_points: int = 512,
    ) -> None:
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.halofit_version = str(halofit_version).strip() or "mead2020"
        self.min_camb_points = int(max(16, min_camb_points))

    @staticmethod
    def _coerce_theta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if arr.shape != (8,):
            raise ValueError(
                "CSST theta must be 8D in order "
                "[Omegab, Omegam, H0, ns, A, w, wa, mnu]."
            )
        if arr[1] <= arr[0]:
            raise ValueError("CSST theta requires Omegam > Omegab.")
        return arr.astype(np.float64)

    @staticmethod
    def _coerce_k_bins(k_bins: Sequence[float] | np.ndarray) -> np.ndarray:
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k.ndim != 1 or k.size <= 0 or np.any(k <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        return k.astype(np.float64)

    def _build_camb_params(self, theta: np.ndarray, k_bins: np.ndarray) -> Any:
        import camb

        omegab, omegam, h0, ns, amp_1e9_as, w0, wa, mnu = (float(value) for value in theta)
        h = h0 / 100.0
        omegac = omegam - omegab
        pars = camb.CAMBparams()
        for attr in ("WantCls", "Want_CMB", "Want_CMB_lensing", "DoLensing"):
            if hasattr(pars, attr):
                setattr(pars, attr, False)
        pars.set_cosmology(
            H0=h0,
            ombh2=omegab * h * h,
            omch2=omegac * h * h,
            mnu=mnu,
        )
        if hasattr(pars, "set_dark_energy"):
            pars.set_dark_energy(w=w0, wa=wa)
        pars.InitPower.set_params(ns=ns, As=amp_1e9_as * 1.0e-9)
        request_kmax = float(max(np.max(k_bins) * 1.25, 0.1))
        pars.set_matter_power(redshifts=[0.0], kmax=request_kmax)
        if hasattr(pars, "NonLinearModel"):
            pars.NonLinearModel.set_params(halofit_version=self.halofit_version)
        elif hasattr(camb, "set_halofit_version"):
            camb.set_halofit_version(self.halofit_version)
        pars.NonLinear = camb.model.NonLinear_both
        return pars

    def get_anchor_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        import camb

        theta_arr = self._coerce_theta(theta)
        k_arr = self._coerce_k_bins(k_bins)
        pars = self._build_camb_params(theta_arr, k_arr)
        results = camb.get_results(pars)
        k_min = float(np.min(k_arr))
        k_max = float(np.max(k_arr))
        camb_minkh = max(k_min * 0.8, 1.0e-5)
        camb_maxkh = max(k_max * 1.2, k_max + 1.0e-6)
        npoints = int(max(self.min_camb_points, k_arr.shape[0]))
        k_camb, _, pk_camb = results.get_matter_power_spectrum(
            minkh=camb_minkh,
            maxkh=camb_maxkh,
            npoints=npoints,
            var1=self.matter_power_var,
            var2=self.matter_power_var,
        )
        if k_min < float(np.min(k_camb)) or k_max > float(np.max(k_camb)):
            raise ValueError(
                "CAMB CSST CDM anchor does not cover requested k range "
                f"[{k_min:.4e}, {k_max:.4e}] with returned range "
                f"[{float(np.min(k_camb)):.4e}, {float(np.max(k_camb)):.4e}]."
            )
        pk = interp1d(k_camb, pk_camb[0], kind="cubic", bounds_error=True)(k_arr)
        return np.maximum(np.asarray(pk, dtype=np.float64), self.power_eps)

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.get_anchor_pk(theta, k_bins)


@dataclass(slots=True)
class CSSTQuijote5DOracle:
    config: Z2Config
    quijote_anchor: CambCDMAnchorProvider = field(init=False)
    csst_provider: Any = field(init=False)

    def __post_init__(self) -> None:
        self.quijote_anchor = CambCDMAnchorProvider(
            reference_as=float(self.config.fastmock_bias.reference_as),
            fixed_w0=float(self.config.fastmock_bias.fixed_w),
            fixed_wa=float(self.config.fastmock_bias.fixed_wa),
            fixed_mnu=float(self.config.fastmock_bias.fixed_mnu),
            power_eps=float(self.config.target.power_eps),
        )
        module = _load_csst_provider_module(self.config)
        provider_cls = module.CSSTDataProvider
        self.csst_provider = provider_cls(
            vendor_path=Path(self.config.fastmock_bias.vendor_path).resolve(),
            checkbound=bool(self.config.fastmock_bias.checkbound),
            anchor_provider=CSSTCDMCAMBAnchorProvider(power_eps=float(self.config.target.power_eps)),
            anchor_provider_name="csst_camb_cdm_hmcode2020",
            truth_backend=str(self.config.fastmock_bias.truth_backend),
            truth_dtype=str(self.config.fastmock_bias.truth_dtype),
            truth_device=str(self.config.fastmock_bias.truth_device),
            power_eps=float(self.config.target.power_eps),
            k_min=1.0e-2,
            k_max=1.0e1,
        )

    def theta5_to_csst8(self, theta_raw: Sequence[float] | np.ndarray, k_bins: np.ndarray) -> np.ndarray:
        return active_to_csst8_theta(
            self.config.parameter_space,
            np.asarray(theta_raw, dtype=np.float64).reshape(1, -1),
            np.asarray(k_bins, dtype=np.float64).reshape(-1),
            self.quijote_anchor,
            fixed_w=float(self.config.fastmock_bias.fixed_w),
            fixed_wa=float(self.config.fastmock_bias.fixed_wa),
            fixed_mnu=float(self.config.fastmock_bias.fixed_mnu),
        )[0]

    def evaluate(
        self,
        theta_raw: np.ndarray,
        k_bins: np.ndarray,
        *,
        prefer_cuda_truth: bool = False,
    ) -> SpectraBatch:
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        theta8 = active_to_csst8_theta(
            self.config.parameter_space,
            theta,
            k,
            self.quijote_anchor,
            fixed_w=float(self.config.fastmock_bias.fixed_w),
            fixed_wa=float(self.config.fastmock_bias.fixed_wa),
            fixed_mnu=float(self.config.fastmock_bias.fixed_mnu),
        )
        truth_backend = (
            str(self.config.fastmock_bias.truth_backend)
            if bool(prefer_cuda_truth)
            else "cpu_batch"
        )
        result = self.csst_provider.run_hifi_residuals(theta8, k, truth_backend=truth_backend)
        log_residual = np.asarray(result["log_residual_hifi"], dtype=np.float64)
        residual_ratio = np.asarray(result["P_residual_hifi"], dtype=np.float64)
        return SpectraBatch(
            theta_raw=theta,
            k_bins=k,
            log_pk=log_residual.astype(np.float64),
            pk=residual_ratio.astype(np.float64),
            metadata={
                "target_kind": "csst_fastmock_official_log_residual",
                "target_transform": "log_csst_nonlin_minus_log_csst_official_hmcode2020_anchor",
                "provider": "csst_official_5d_adapter",
                "pk_semantics": "residual_ratio_not_original_power",
                "anchor_power_provider": "csst_official_hmcode2020",
                "anchor_matches_official_get_pknl": True,
                "csst_theta8": theta8.astype(np.float64),
                "csst_truth_batch_vectorized": bool(result.get("metadata", {}).get("batch_vectorized", False)),
                "csst_truth_backend_used": str(result.get("metadata", {}).get("truth_backend_used", truth_backend)),
                "fixed_params": {
                    "w": float(self.config.fastmock_bias.fixed_w),
                    "wa": float(self.config.fastmock_bias.fixed_wa),
                    "mnu": float(self.config.fastmock_bias.fixed_mnu),
                },
            },
        )


@dataclass(slots=True)
class CSSTBiasModel:
    config: Z2Config
    oracle: CSSTQuijote5DOracle
    emulator: PCAGPDirectCDMEmulator
    k_bins: np.ndarray
    normalization: str
    normalization_probe_count: int
    score_mode: str
    bias_weight: float
    bias_band_weights: np.ndarray
    bias_k_weights: np.ndarray
    bias_weight_details: dict[str, Any]
    cache_decimals: int
    enabled: bool = True
    _cache: dict[tuple[float, ...], float] = field(default_factory=dict)

    def bias_for_unit(self, unit_points: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit_points, dtype=np.float64)
        if unit.ndim == 1:
            unit = unit.reshape(1, -1)
        theta = self.config.parameter_space.denormalize(np.clip(unit, 0.0, 1.0))
        return self.bias(theta)

    def bias_for_unit_batch(
        self,
        unit_points: np.ndarray,
        *,
        chunk_size: int = 6144,
        prefer_cuda_truth: bool = True,
    ) -> np.ndarray:
        unit = np.asarray(unit_points, dtype=np.float64)
        if unit.ndim == 1:
            unit = unit.reshape(1, -1)
        if unit.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        batch = int(max(1, min(int(chunk_size), int(self.config.fastmock_bias.truth_chunk_size))))
        values: list[np.ndarray] = []
        start = 0
        while start < unit.shape[0]:
            stop = min(start + batch, unit.shape[0])
            try:
                theta = self.config.parameter_space.denormalize(np.clip(unit[start:stop], 0.0, 1.0))
                values.append(self._bias_uncached(theta, prefer_cuda_truth=bool(prefer_cuda_truth)))
                start = stop
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or batch <= 1 or not bool(prefer_cuda_truth):
                    raise
                batch = max(1, batch // 2)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return np.concatenate(values, axis=0).astype(np.float64)

    def _bias_uncached(self, theta_raw: np.ndarray, *, prefer_cuda_truth: bool = False) -> np.ndarray:
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        pred = self.emulator.predict(theta)
        truth = self.oracle.evaluate(theta, self.k_bins, prefer_cuda_truth=bool(prefer_cuda_truth))
        relative = np.abs(np.exp(pred.log_pk_mean - truth.log_pk) - 1.0)
        values = relative @ np.asarray(self.bias_k_weights, dtype=np.float64).reshape(-1)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def bias(self, theta_raw: np.ndarray, *, prefer_cuda_truth: bool = False) -> np.ndarray:
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        out = np.empty((theta.shape[0],), dtype=np.float64)
        missing_rows: list[np.ndarray] = []
        missing_indices: list[int] = []
        missing_keys: list[tuple[float, ...]] = []
        for index, row in enumerate(theta):
            key = tuple(np.round(row.astype(np.float64), int(self.cache_decimals)).tolist())
            cached = self._cache.get(key)
            if cached is None:
                missing_rows.append(row)
                missing_indices.append(index)
                missing_keys.append(key)
            else:
                out[index] = float(cached)
        if missing_rows:
            missing_theta = np.vstack(missing_rows).astype(np.float64)
            values = self._bias_uncached(missing_theta, prefer_cuda_truth=bool(prefer_cuda_truth))
            for index, key, value in zip(missing_indices, missing_keys, values.tolist(), strict=True):
                value_float = float(max(value, 0.0))
                self._cache[key] = value_float
                out[index] = value_float
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def fit_csst_bias_model(
    *,
    config: Z2Config,
    train_theta_raw: np.ndarray,
    k_bins: np.ndarray,
) -> CSSTBiasModel:
    oracle = CSSTQuijote5DOracle(config)
    labels = oracle.evaluate(train_theta_raw, k_bins)
    emulator = PCAGPDirectCDMEmulator(
        config.parameter_space,
        config.model,
        target_kind="csst_fastmock_official_log_residual",
    ).fit(
        np.asarray(train_theta_raw, dtype=np.float64),
        labels.log_pk,
        k_bins,
    )
    bias_band_weights = _resolve_bias_band_weights(config)
    bias_k_weights, bias_weight_details = _build_bias_k_weights(
        config,
        np.asarray(k_bins, dtype=np.float64).reshape(-1),
        bias_band_weights,
    )
    return CSSTBiasModel(
        config=config,
        oracle=oracle,
        emulator=emulator,
        k_bins=np.asarray(k_bins, dtype=np.float64).reshape(-1),
        normalization=str(config.fastmock_bias.normalization).strip().lower(),
        normalization_probe_count=int(config.fastmock_bias.normalization_probe_count),
        score_mode=str(config.fastmock_bias.score_mode).strip().lower(),
        bias_weight=float(config.fastmock_bias.bias_weight),
        bias_band_weights=bias_band_weights.astype(np.float64),
        bias_k_weights=bias_k_weights.astype(np.float64),
        bias_weight_details=dict(bias_weight_details),
        cache_decimals=int(config.fastmock_bias.cache_decimals),
    )


def _load_csst_provider_module(config: Z2Config) -> Any:
    del config
    return csst_data_provider


def _resolve_bias_band_weights(config: Z2Config) -> np.ndarray:
    configured = tuple(float(value) for value in config.fastmock_bias.bias_band_weights)
    if configured:
        return np.asarray(configured, dtype=np.float64)
    return np.ones((len(config.k_grid.bands),), dtype=np.float64)


def _build_bias_k_weights(
    config: Z2Config,
    k_bins: np.ndarray,
    band_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    integration_weights = _logk_trapezoid_weights(k_arr)
    curve, band_curve_means = _smooth_logk_bias_weight_curve(config, k_arr, band_weights)
    raw = np.maximum(integration_weights * curve, 0.0)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        point_weights = np.full((k_arr.shape[0],), 1.0 / float(k_arr.shape[0]), dtype=np.float64)
    else:
        point_weights = (raw / total).astype(np.float64)
    return point_weights.astype(np.float64), {
        "mode": "smooth_logk_weighted_relative_error",
        "bias_band_weights": np.asarray(band_weights, dtype=np.float64).tolist(),
        "bias_k_weight_sum": float(np.sum(point_weights)),
        "bias_k_weight_min": float(np.min(point_weights)) if point_weights.size else 0.0,
        "bias_k_weight_max": float(np.max(point_weights)) if point_weights.size else 0.0,
        "bias_k_weight_curve_band_means": band_curve_means.astype(np.float64).tolist(),
        "bias_statistic": "weighted_mean_relative_error_over_k",
    }


def _smooth_logk_bias_weight_curve(
    config: Z2Config,
    k_bins: np.ndarray,
    band_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    weights = np.asarray(band_weights, dtype=np.float64).reshape(-1)
    band_count = len(config.k_grid.bands)
    if weights.shape != (band_count,):
        weights = np.ones((band_count,), dtype=np.float64)
    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    width = float(max(config.active_learning.pca_weight_transition_dex, 1.0e-4))
    curve = np.full_like(k_arr, max(float(weights[0]), 1.0e-12), dtype=np.float64)
    for band_index, band in enumerate(config.k_grid.bands[:-1]):
        boundary = np.log10(float(band.k_max))
        next_level = max(float(weights[band_index + 1]), 1.0e-12)
        smooth_step = 0.5 * (1.0 + np.tanh((logk - boundary) / width))
        curve = (1.0 - smooth_step) * curve + smooth_step * next_level
    integration_weights = _logk_trapezoid_weights(k_arr)
    curve_mean = float(
        np.sum(curve * integration_weights) / max(np.sum(integration_weights), 1.0e-30)
    )
    curve = curve / max(curve_mean, 1.0e-30)

    band_means: list[float] = []
    for index, band in enumerate(config.k_grid.bands):
        if index == band_count - 1:
            mask = (k_arr >= float(band.k_min)) & (k_arr <= float(band.k_max))
        else:
            mask = (k_arr >= float(band.k_min)) & (k_arr < float(band.k_max))
        if not np.any(mask):
            band_means.append(0.0)
            continue
        local_weights = integration_weights[mask]
        band_means.append(
            float(np.sum(curve[mask] * local_weights) / max(np.sum(local_weights), 1.0e-30))
        )
    return curve.astype(np.float64), np.asarray(band_means, dtype=np.float64)


def _logk_trapezoid_weights(k_bins: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if k_arr.ndim != 1 or k_arr.size < 2:
        raise ValueError("k_bins must be a 1D array with at least two points.")
    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    weights = np.empty_like(logk, dtype=np.float64)
    weights[0] = 0.5 * (logk[1] - logk[0])
    weights[-1] = 0.5 * (logk[-1] - logk[-2])
    if logk.size > 2:
        weights[1:-1] = 0.5 * (logk[2:] - logk[:-2])
    return np.maximum(weights, 0.0).astype(np.float64)
