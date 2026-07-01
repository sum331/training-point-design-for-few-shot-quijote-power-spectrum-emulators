from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.interpolate import interp1d

from .config import Z2Config
from .runtime_core.quijote_gp_surrogate import load_quijote_gp_surrogate
from .theta_transform import active_to_quijote_theta


@dataclass(frozen=True, slots=True)
class SpectraBatch:
    theta_raw: np.ndarray
    k_bins: np.ndarray
    log_pk: np.ndarray
    pk: np.ndarray
    metadata: dict[str, Any]


class DirectCDMOracle(Protocol):
    target_kind: str

    def evaluate(self, theta_raw: np.ndarray, k_bins: np.ndarray) -> SpectraBatch:
        ...


class CambCDMAnchorProvider:
    """CAMB/HMCODE2020 CDM nonlinear anchor on the Quijote 5D slice."""

    provider_name = "camb_cdm_hmcode2020_sigma8_calibrated"
    anchor_mode = "camb_cdm_hmcode2020"
    power_label = "cdm_hmcode2020"
    matter_power_var = "delta_cdm"

    def __init__(
        self,
        *,
        reference_as: float = 2.1e-9,
        halofit_version: str = "mead2020",
        fixed_w0: float = -1.0,
        fixed_wa: float = 0.0,
        fixed_mnu: float = 0.0,
        power_eps: float = 1.0e-12,
        min_camb_points: int = 512,
    ) -> None:
        self.reference_as = float(reference_as)
        self.halofit_version = str(halofit_version).strip() or "mead2020"
        self.fixed_w0 = float(fixed_w0)
        self.fixed_wa = float(fixed_wa)
        self.fixed_mnu = float(fixed_mnu)
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.min_camb_points = int(max(16, min_camb_points))
        self._sigma8_cache: dict[tuple[float, ...], float] = {}
        self._anchor_cache: dict[tuple[float, ...], np.ndarray] = {}

    @staticmethod
    def coerce_theta(theta: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape != (5,):
            raise ValueError(f"Quijote theta must have shape [5], got {theta_arr.shape}.")
        omega_m, omega_b, h, _, sigma_8 = (float(value) for value in theta_arr)
        if omega_m <= omega_b:
            raise ValueError(
                "Quijote theta is invalid for CAMB CDM anchor: Omega_m must exceed "
                f"Omega_b, got Omega_m={omega_m:.6g}, Omega_b={omega_b:.6g}."
            )
        if h <= 0.0 or sigma_8 <= 0.0:
            raise ValueError("Quijote theta is invalid for CAMB CDM anchor: h and sigma_8 must be positive.")
        return theta_arr.astype(np.float64)

    @staticmethod
    def coerce_k_bins(k_bins: np.ndarray) -> np.ndarray:
        k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_arr.ndim != 1 or k_arr.size <= 0 or np.any(k_arr <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        return k_arr.astype(np.float64)

    def _build_camb_params(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
        nonlinear: bool,
    ) -> Any:
        import camb

        omega_m, omega_b, h, n_s, _ = (float(value) for value in theta)
        omega_cdm = omega_m - omega_b
        pars = camb.CAMBparams()
        for attr in ("WantCls", "Want_CMB", "Want_CMB_lensing", "DoLensing"):
            if hasattr(pars, attr):
                setattr(pars, attr, False)
        pars.set_cosmology(
            H0=100.0 * h,
            ombh2=omega_b * h * h,
            omch2=omega_cdm * h * h,
            mnu=self.fixed_mnu,
        )
        if hasattr(pars, "set_dark_energy"):
            pars.set_dark_energy(w=self.fixed_w0, wa=self.fixed_wa)
        pars.InitPower.set_params(ns=n_s, As=float(primordial_as))
        request_kmax = float(max(np.max(k_bins) * 1.25, 0.1))
        pars.set_matter_power(redshifts=[0.0], kmax=request_kmax)
        if hasattr(pars, "NonLinearModel"):
            pars.NonLinearModel.set_params(halofit_version=self.halofit_version)
        elif hasattr(camb, "set_halofit_version"):
            camb.set_halofit_version(self.halofit_version)
        pars.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
        return pars

    def sigma8_for_as(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
    ) -> float:
        import camb

        theta_arr = self.coerce_theta(theta)
        k_arr = self.coerce_k_bins(k_bins)
        key = self._sigma8_cache_key(theta_arr, k_arr, primordial_as=float(primordial_as))
        cached = self._sigma8_cache.get(key)
        if cached is not None:
            return float(cached)
        pars = self._build_camb_params(
            theta_arr,
            k_arr,
            primordial_as=primordial_as,
            nonlinear=False,
        )
        results = camb.get_results(pars)
        sigma8_values = np.asarray(results.get_sigma8(), dtype=np.float64).reshape(-1)
        if sigma8_values.size <= 0 or not np.isfinite(sigma8_values[0]) or sigma8_values[0] <= 0.0:
            raise RuntimeError("CAMB returned an invalid sigma8 during CDM anchor calibration.")
        value = float(sigma8_values[0])
        self._sigma8_cache[key] = value
        return value

    def target_as_for_sigma8(self, theta: np.ndarray, k_bins: np.ndarray) -> float:
        theta_arr = self.coerce_theta(theta)
        k_arr = self.coerce_k_bins(k_bins)
        target_sigma8 = float(theta_arr[4])
        reference_sigma8 = self.sigma8_for_as(
            theta_arr,
            k_arr,
            primordial_as=self.reference_as,
        )
        return float(self.reference_as * (target_sigma8 / reference_sigma8) ** 2)

    def _hmcode_pk_for_as(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
    ) -> np.ndarray:
        import camb

        pars = self._build_camb_params(
            theta,
            k_bins,
            primordial_as=primordial_as,
            nonlinear=True,
        )
        results = camb.get_results(pars)
        k_min = float(np.min(k_bins))
        k_max = float(np.max(k_bins))
        camb_minkh = max(k_min * 0.8, 1.0e-5)
        camb_maxkh = max(k_max * 1.2, k_max + 1.0e-6)
        npoints = int(max(self.min_camb_points, k_bins.shape[0]))
        k_camb, _, pk_camb = results.get_matter_power_spectrum(
            minkh=camb_minkh,
            maxkh=camb_maxkh,
            npoints=npoints,
            var1=self.matter_power_var,
            var2=self.matter_power_var,
        )
        if float(np.min(k_bins)) < float(np.min(k_camb)) or float(np.max(k_bins)) > float(np.max(k_camb)):
            raise ValueError(
                "CAMB CDM anchor does not cover requested Quijote k range "
                f"[{k_min:.4e}, {k_max:.4e}] with returned range "
                f"[{float(np.min(k_camb)):.4e}, {float(np.max(k_camb)):.4e}]."
            )
        hmcode = interp1d(k_camb, pk_camb[0], kind="cubic", bounds_error=True)(k_bins)
        return np.maximum(np.asarray(hmcode, dtype=np.float64), self.power_eps)

    def get_anchor_pk(self, theta: np.ndarray, k_bins: np.ndarray) -> np.ndarray:
        theta_arr = self.coerce_theta(theta)
        k_arr = self.coerce_k_bins(k_bins)
        key = self._anchor_cache_key(theta_arr, k_arr)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached.copy()
        target_as = self.target_as_for_sigma8(theta_arr, k_arr)
        anchor = self._hmcode_pk_for_as(theta_arr, k_arr, primordial_as=target_as)
        self._anchor_cache[key] = anchor.astype(np.float64)
        return anchor

    def get_linear_pk(self, theta: np.ndarray, k_bins: np.ndarray) -> np.ndarray:
        return self.get_anchor_pk(theta, k_bins)

    def _sigma8_cache_key(self, theta: np.ndarray, k_bins: np.ndarray, *, primordial_as: float) -> tuple[float, ...]:
        omega_m, omega_b, h, n_s, _ = (float(value) for value in theta)
        return (
            round(omega_m, 10),
            round(omega_b, 10),
            round(h, 10),
            round(n_s, 10),
            round(float(primordial_as), 18),
            float(k_bins.shape[0]),
            round(float(k_bins[0]), 10),
            round(float(k_bins[-1]), 10),
            round(float(np.mean(np.log(k_bins))), 10),
        )

    def _anchor_cache_key(self, theta: np.ndarray, k_bins: np.ndarray) -> tuple[float, ...]:
        return tuple(round(float(value), 10) for value in theta.tolist()) + (
            float(k_bins.shape[0]),
            round(float(k_bins[0]), 10),
            round(float(k_bins[-1]), 10),
            round(float(np.mean(np.log(k_bins))), 10),
        )


class V2TruthGeneratorOracle:
    target_kind = "direct_cdm_logpk"

    def __init__(self, config: Z2Config) -> None:
        self.config = config
        self._surrogate: Any | None = None
        self._coordinate_anchor: CambCDMAnchorProvider | None = None

    def evaluate(self, theta_raw: np.ndarray, k_bins: np.ndarray) -> SpectraBatch:
        theta = np.asarray(theta_raw, dtype=np.float64)
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[1] != self.config.parameter_space.dim:
            raise ValueError(f"theta_raw must have shape [N,{self.config.parameter_space.dim}], got {theta.shape}.")
        if np.any(k <= 0.0) or np.any(np.diff(k) <= 0.0):
            raise ValueError("k_bins must be strictly increasing and positive.")
        quijote_theta = active_to_quijote_theta(
            self.config.parameter_space,
            theta,
            k,
            self._get_coordinate_anchor(),
        )
        chunks: list[np.ndarray] = []
        pk_chunks: list[np.ndarray] = []
        chunk_size = max(1, int(self.config.resources.truth_generator.chunk_size))
        surrogate = self._load_surrogate()
        device = self.config.resources.truth_generator.device
        if str(device).strip().lower() == "auto":
            device = "cuda"
        requested_device = str(device)
        devices_used: list[str] = []
        fallback_errors: list[str] = []
        for start in range(0, theta.shape[0], chunk_size):
            block = quijote_theta[start : start + chunk_size]
            try:
                prediction = surrogate.predict(
                    block,
                    input_space="raw",
                    k_target=k,
                    return_std=False,
                    device=device,
                )
                devices_used.append(str(device))
            except Exception:
                if str(device).lower().startswith("cuda"):
                    fallback_errors.append("cuda prediction failed; fell back to cpu")
                    prediction = surrogate.predict(
                        block,
                        input_space="raw",
                        k_target=k,
                        return_std=False,
                        device="cpu",
                    )
                    devices_used.append("cpu")
                else:
                    raise
            chunks.append(np.asarray(prediction["log_pk_mean"], dtype=np.float64))
            pk_chunks.append(np.asarray(prediction["pk_mean"], dtype=np.float64))
        log_pk = np.vstack(chunks).astype(np.float64)
        pk = np.vstack(pk_chunks).astype(np.float64)
        return SpectraBatch(
            theta_raw=theta,
            k_bins=k,
            log_pk=log_pk,
            pk=pk,
            metadata={
                "target_kind": self.target_kind,
                "oracle": "v2_direct_logpk_truth_generator",
                "truth_generator_path": str(self.config.resources.truth_generator.path),
                "truth_generator_requested_device": requested_device,
                "truth_generator_devices_used": sorted(set(devices_used)),
                "truth_generator_cuda_fallback_count": len(fallback_errors),
                "active_parameter_space": str(self.config.parameter_space.name),
                "quijote_theta_raw": quijote_theta.astype(np.float64),
            },
        )

    def _get_coordinate_anchor(self) -> CambCDMAnchorProvider:
        if self._coordinate_anchor is None:
            self._coordinate_anchor = CambCDMAnchorProvider(power_eps=float(self.config.target.power_eps))
        return self._coordinate_anchor

    def _load_surrogate(self) -> Any:
        if self._surrogate is not None:
            return self._surrogate
        path = self.config.resources.truth_generator.path
        if not path.exists():
            raise FileNotFoundError(f"z2 direct-logP truth generator not found: {path}")

        loaded = load_quijote_gp_surrogate(path)
        metadata = dict(getattr(loaded, "metadata", {}) or {})
        target_transform = str(metadata.get("target_transform", metadata.get("training_target", "")))
        training_target = str(metadata.get("training_target", target_transform))
        if target_transform != "direct_logpk" or training_target != "direct_logpk":
            raise ValueError(
                "The configured z2 truth generator is not direct_logpk; "
                f"target_transform={target_transform!r}, training_target={training_target!r}."
            )
        if bool(metadata.get("linear_anchor_inside_generator", False)):
            raise ValueError("z2 requires a direct CDM generator with no linear/HMCode anchor inside.")
        self._surrogate = loaded
        return loaded


class SyntheticDirectCDMOracle:
    target_kind = "direct_cdm_logpk"

    def __init__(self, config: Z2Config) -> None:
        self.config = config

    def evaluate(self, theta_raw: np.ndarray, k_bins: np.ndarray) -> SpectraBatch:
        theta = np.asarray(theta_raw, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        unit = self.config.parameter_space.normalize(theta)
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        logk = np.log(k).reshape(1, -1)
        amp = 3.0 + 0.8 * unit[:, [0]] - 0.25 * unit[:, [1]] + 0.35 * unit[:, [4]]
        tilt = -1.15 + 0.28 * (unit[:, [3]] - 0.5)
        wiggle = 0.06 * np.sin(4.0 * logk + 2.0 * unit[:, [2]])
        bend = -0.18 * (unit[:, [0]] - unit[:, [4]]) * (k.reshape(1, -1) / 3.0) ** 0.7
        log_pk = amp + tilt * logk + wiggle + bend
        pk = np.exp(log_pk)
        return SpectraBatch(
            theta_raw=theta,
            k_bins=k,
            log_pk=log_pk.astype(np.float64),
            pk=pk.astype(np.float64),
            metadata={"target_kind": self.target_kind, "oracle": "synthetic_direct_cdm"},
        )


class LogdiffCDMOracle:
    def __init__(self, config: Z2Config, base_oracle: DirectCDMOracle, anchor_provider: CambCDMAnchorProvider) -> None:
        self.config = config
        self.base_oracle = base_oracle
        self.anchor_provider = anchor_provider
        self.target_kind = str(config.target.kind)

    def evaluate(self, theta_raw: np.ndarray, k_bins: np.ndarray) -> SpectraBatch:
        base = self.base_oracle.evaluate(theta_raw, k_bins)
        quijote_theta = active_to_quijote_theta(
            self.config.parameter_space,
            base.theta_raw,
            base.k_bins,
            self.anchor_provider,
        )
        anchors = compute_anchor_batch(
            self.anchor_provider,
            quijote_theta,
            base.k_bins,
            power_eps=self.config.target.power_eps,
        )
        logdiff = np.log(np.maximum(base.pk, self.config.target.power_eps)) - np.log(
            np.maximum(anchors, self.config.target.power_eps)
        )
        return SpectraBatch(
            theta_raw=base.theta_raw,
            k_bins=base.k_bins,
            log_pk=logdiff.astype(np.float64),
            pk=base.pk.astype(np.float64),
            metadata={
                "target_kind": self.target_kind,
                "target_transform": target_transform_name(self.config),
                "oracle": "v2_direct_logpk_truth_generator_plus_camb_cdm_logdiff",
                "base_oracle_metadata": dict(base.metadata),
                "anchor_provider": self.anchor_provider.provider_name,
                "anchor_power_label": self.anchor_provider.power_label,
                "quijote_theta_raw": quijote_theta.astype(np.float64),
            },
        )


def compute_anchor_batch(
    anchor_provider: CambCDMAnchorProvider,
    theta_raw: np.ndarray,
    k_bins: np.ndarray,
    *,
    power_eps: float,
) -> np.ndarray:
    theta = np.asarray(theta_raw, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    rows = [
        np.asarray(anchor_provider.get_anchor_pk(row, k), dtype=np.float64).reshape(1, -1)
        for row in theta
    ]
    if not rows:
        return np.empty((0, k.shape[0]), dtype=np.float64)
    anchors = np.vstack(rows).astype(np.float64)
    return np.maximum(anchors, float(max(power_eps, 1.0e-30)))


def make_anchor_provider(config: Z2Config, *, reference_as: float | None = None) -> CambCDMAnchorProvider:
    if str(config.target.kind).strip().lower() != "cdm_logdiff":
        raise ValueError("CAMB CDM anchor provider is only valid for target.kind=cdm_logdiff.")
    return CambCDMAnchorProvider(
        reference_as=float(reference_as if reference_as is not None else 2.1e-9),
        power_eps=float(config.target.power_eps),
    )


def target_transform_name(config: Z2Config) -> str:
    if str(config.target.kind).strip().lower() == "direct_cdm_logpk":
        return "direct_logpk"
    anchor = str(config.target.anchor_mode).strip().lower() or "camb_cdm_hmcode2020"
    return f"log_hi_minus_log_{anchor}_anchor"


def make_oracle(config: Z2Config) -> DirectCDMOracle:
    if config.oracle_kind == "synthetic_direct_cdm":
        return SyntheticDirectCDMOracle(config)
    direct = V2TruthGeneratorOracle(config)
    if str(config.target.kind).strip().lower() == "cdm_logdiff":
        return LogdiffCDMOracle(config, direct, make_anchor_provider(config))
    return direct
