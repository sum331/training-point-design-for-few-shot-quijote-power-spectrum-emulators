"""CAMB 数据提供器。

本文件对应以下文档职责：
- `新的验证大纲_正式交付版.md` 中的 `CAMB` 主数据源接口
- `实施清单与API接口清单.md` 中的 `get_linear_pk`、`run_lofi_cloud`、`run_hifi_anchor`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng

from z2quijote.runtime_core.config import ValidationRuntimeConfig
from z2quijote.runtime_core.types import ArrayLike

_REAL_CAMB_CALL_COUNTER = 0
K_RANGE_MIN = 1.0e-2
K_RANGE_MAX = 1.0e1


def _is_hifi_mode(mode: str) -> bool:
    mode_norm = str(mode).strip().lower()
    return mode_norm in {
        "hifi",
        "placeholder-hifi",
        "real-hifi",
        "hifi_highk_enhanced",
        "placeholder-hifi_highk_enhanced",
        "real-hifi_highk_enhanced",
    }


def _is_hifi_highk_mode(mode: str) -> bool:
    mode_norm = str(mode).strip().lower()
    return mode_norm in {
        "hifi_highk_enhanced",
        "placeholder-hifi_highk_enhanced",
        "real-hifi_highk_enhanced",
    }


def _display_columns(text: str) -> int:
    cols = 0
    for ch in text:
        cols += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return cols


def _truncate_to_columns(text: str, max_cols: int) -> str:
    if max_cols <= 0:
        return ""
    out: list[str] = []
    cols = 0
    for ch in text:
        ch_cols = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
        if cols + ch_cols > max_cols:
            break
        out.append(ch)
        cols += ch_cols
    return "".join(out)


@dataclass(slots=True)
class CAMBAccuracyConfig:
    mode: str
    amplitude_scale: float = 1.0
    smoothness_scale: float = 1.0
    notes: str = ""

@dataclass(slots=True)
class CAMBNoiseConfig:
    inject_white_noise: bool = True
    inject_outliers: bool = True
    corrupted_ratio: float = 0.05
    white_noise_scale: float | None = None
    outlier_multiplier: float = 1.25
    outlier_bias: float = 0.0

@dataclass(slots=True)
class CAMBPlaceholderWarning:
    backend_name: str
    placeholder_active: bool
    message: str
    backend_used: str
    spectrum_type: str
    lofi_strategy: str

@dataclass(slots=True)
class CAMBDataProvider:
    config: ValidationRuntimeConfig
    rng: Generator | None = None
    _last_progress_key: tuple[str, ...] | None = None
    _last_progress_time: float = 0.0

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = default_rng(self.config.camb.placeholder_noise_seed)

    def _lofi_compute_method_label(self, *, ratio_transfer_applied: bool, tier: str) -> str:
        if self.config.camb.allow_placeholder_backend:
            return "物理近似(placeholder)"
        if ratio_transfer_applied:
            if tier == "L3":
                return "物理近似(L3)"
            return f"CAMB线性+比值迁移({tier})"
        return "CAMB计算"

    def _report_lofi_point_progress(
        self,
        current: int,
        total: int,
        method_label: str,
        *,
        progress_context: dict[str, int | str] | None = None,
        force_newline: bool = False,
    ) -> None:
        cloud_text = f"第 {current}/{total} 个云点"
        if progress_context:
            cycle_text = ""
            if progress_context.get("cycle_current") and progress_context.get("cycle_total"):
                cycle_text = (
                    f"第 {progress_context['cycle_current']}/{progress_context['cycle_total']} 个循环，"
                )
            iter_text = ""
            if progress_context.get("iteration_current") and progress_context.get("iteration_total"):
                iter_text = (
                    f"第 {progress_context['iteration_current']}/{progress_context['iteration_total']} 次迭代，"
                )
            anchor_text = ""
            if progress_context.get("anchor_current") and progress_context.get("anchor_total"):
                anchor_text = (
                    f"第 {progress_context['anchor_current']}/{progress_context['anchor_total']} 个锚点，"
                )
            lofi_total_text = ""
            if progress_context.get("anchor_current") and progress_context.get("anchor_total"):
                try:
                    anchor_current = int(progress_context["anchor_current"])
                    anchor_total = int(progress_context["anchor_total"])
                    lofi_total = int(anchor_total * total)
                    lofi_done = int((anchor_current - 1) * total + current)
                    lofi_total_text = f"Lo-Fi点 {lofi_done}/{lofi_total}，"
                except Exception:
                    lofi_total_text = ""
            phase = str(progress_context.get("phase", "")).strip()
            phase_text = f"[{phase}] " if phase else ""
            line = (
                f"{phase_text}{cycle_text}{iter_text}{anchor_text}"
                f"{lofi_total_text}{cloud_text} ({method_label})"
            )
        else:
            line = f"{cloud_text} ({method_label})"
        # 进度去刷屏：同一阶段（phase/cycle/iteration）仅输出一次摘要。
        progress_key = (
            str(progress_context.get("phase", "")) if progress_context else "",
            str(progress_context.get("cycle_current", "")) if progress_context else "",
            str(progress_context.get("iteration_current", "")) if progress_context else "",
            str(method_label),
        )
        if self._last_progress_key == progress_key:
            return
        self._last_progress_key = progress_key
        summary_parts: list[str] = ["[计算]"]
        if progress_context:
            phase = str(progress_context.get("phase", "")).strip()
            if phase:
                summary_parts.append(f"[{phase}]")
            if progress_context.get("cycle_current") and progress_context.get("cycle_total"):
                summary_parts.append(
                    f"循环 {progress_context['cycle_current']}/{progress_context['cycle_total']}"
                )
            if progress_context.get("iteration_current") and progress_context.get("iteration_total"):
                summary_parts.append(
                    f"迭代 {progress_context['iteration_current']}/{progress_context['iteration_total']}"
                )
        summary_parts.append(f"Lo-Fi 计算进行中（{method_label}）")
        print(" ".join(summary_parts), flush=True)
        return

    def _coerce_theta(self, theta: ArrayLike) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=np.float64)
        if theta_array.ndim != 1:
            raise ValueError(f"theta must have shape [d], got {theta_array.shape}.")
        return theta_array

    def _coerce_thetas(self, thetas: ArrayLike) -> np.ndarray:
        theta_array = np.asarray(thetas, dtype=np.float64)
        if theta_array.ndim != 2:
            raise ValueError(f"thetas must have shape [N, d], got {theta_array.shape}.")
        return theta_array

    def _coerce_k_bins(self, k_bins: ArrayLike) -> np.ndarray:
        k_array = np.asarray(k_bins, dtype=np.float64)
        if k_array.ndim != 1:
            raise ValueError(f"k_bins must have shape [N_k], got {k_array.shape}.")
        if np.any(k_array <= 0.0):
            raise ValueError("k_bins must be strictly positive.")
        k_min = float(np.min(k_array))
        k_max = float(np.max(k_array))
        if k_min < K_RANGE_MIN or k_max > K_RANGE_MAX:
            raise ValueError(
                f"k_bins must stay within [{K_RANGE_MIN:.1e}, {K_RANGE_MAX:.1e}] h/Mpc, "
                f"got [{k_min:.4e}, {k_max:.4e}]."
            )
        return k_array

    def _backend_warning(self) -> CAMBPlaceholderWarning:
        placeholder_active = self.config.camb.allow_placeholder_backend
        backend_used = "placeholder" if placeholder_active else self.config.camb.backend_name
        return CAMBPlaceholderWarning(
            backend_name=self.config.camb.backend_name,
            placeholder_active=placeholder_active,
            message=(
                "Placeholder CAMB backend is active. "
                "These spectra are for scaffold integration only and must not "
                "be used as final validation evidence."
                if placeholder_active
                else ""
            ),
            backend_used=backend_used,
            spectrum_type=self.config.camb.spectrum_type,
            lofi_strategy=self.config.camb.lofi_strategy,
        )

    def _resolve_accuracy_config(self, accuracy_config: CAMBAccuracyConfig) -> CAMBAccuracyConfig:
        mode = accuracy_config.mode.lower()
        if mode in {"linear", "placeholder-linear"}:
            return CAMBAccuracyConfig(mode=accuracy_config.mode, amplitude_scale=1.0, smoothness_scale=1.0)
        if mode in {"lofi", "placeholder-lofi"}:
            return CAMBAccuracyConfig(mode=accuracy_config.mode, amplitude_scale=0.98, smoothness_scale=0.95)
        if _is_hifi_mode(mode):
            return CAMBAccuracyConfig(mode=accuracy_config.mode, amplitude_scale=1.02, smoothness_scale=1.05)
        return accuracy_config

    def _placeholder_linear_pk(self, theta: np.ndarray, k_bins: np.ndarray, accuracy_config: CAMBAccuracyConfig) -> np.ndarray:
        resolved = self._resolve_accuracy_config(accuracy_config)
        theta_scale = 1.0 + 0.01 * np.tanh(float(np.mean(theta)))
        slope = 1.0 + 0.05 * np.tanh(float(np.std(theta)))
        return resolved.amplitude_scale * theta_scale / (1.0 + np.power(k_bins, 1.2 * resolved.smoothness_scale * slope))

    def _placeholder_nonlinear_pk(self, theta, k_bins, accuracy_config):
        resolved = self._resolve_accuracy_config(accuracy_config)
        linear_pk = self._placeholder_linear_pk(theta, k_bins, resolved)
        curvature = 0.12 + 0.02 * np.tanh(theta[0])
        ratio = 1.0 + resolved.smoothness_scale * curvature * np.exp(-0.5 * k_bins)
        if _is_hifi_mode(resolved.mode):
            ratio = ratio + self.config.camb.placeholder_hifi_boost
        return np.maximum(linear_pk * ratio, self.config.eps_r)

    def _apply_controlled_noise(self, spectra, noise_config):
        if not noise_config.inject_white_noise:
            return spectra
        noise_scale = (
            self.config.camb.placeholder_lofi_noise_scale
            if noise_config.white_noise_scale is None
            else float(noise_config.white_noise_scale)
        )
        noise = self.rng.normal(loc=0.0, scale=noise_scale, size=spectra.shape)
        return np.maximum(spectra * (1.0 + noise), self.config.eps_r)

    def _apply_controlled_outliers(self, spectra, noise_config):
        if not noise_config.inject_outliers or spectra.shape[0] == 0:
            return spectra, 0
        corrupted_count = int(round(spectra.shape[0] * noise_config.corrupted_ratio))
        corrupted_count = min(corrupted_count, spectra.shape[0])
        if corrupted_count <= 0:
            return spectra, 0
        result = spectra.copy()
        row_indices = self.rng.choice(spectra.shape[0], size=corrupted_count, replace=False)
        col_indices = self.rng.integers(0, spectra.shape[1], size=corrupted_count)
        result[row_indices, col_indices] = np.maximum(
            result[row_indices, col_indices] * noise_config.outlier_multiplier + noise_config.outlier_bias,
            self.config.eps_r,
        )
        return result, corrupted_count

    def _call_real_camb(self, theta, k_bins, is_lofi=False, only_linear=False, accuracy_config: CAMBAccuracyConfig | None = None):
        global _REAL_CAMB_CALL_COUNTER
        _REAL_CAMB_CALL_COUNTER += 1
        import camb
        from scipy.interpolate import interp1d
        
        Omegab, Omegacb, H0, ns, A, w0, wa, sum_m_nu = theta
        h = float(H0) / 100.0
        ombh2 = float(Omegab) * (h**2)
        omch2 = float(Omegacb) * (h**2) - ombh2
        if omch2 <= 0.0:
            raise ValueError(
                "Invalid cosmology parameters: derived omch2 must be positive, "
                f"got {omch2:.6e} from theta={np.asarray(theta, dtype=np.float64).tolist()}."
            )
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_m_nu)
        use_ppf = bool(wa != 0.0 and (w0 < -1.0 or (1.0 + w0 + wa) < 0.0))
        dark_energy_model = "ppf" if use_ppf else "fluid"
        try:
            pars.set_dark_energy(w=w0, wa=wa, dark_energy_model=dark_energy_model)
        except Exception:
            raise
        mode = "" if accuracy_config is None else str(accuracy_config.mode).strip().lower()
        use_highk_hifi = bool((not is_lofi) and (_is_hifi_highk_mode(mode) or (mode in {"", "hifi"} and bool(self.config.camb.camb_hifi_highk_enabled))))
        request_kmax = float(np.max(k_bins) * 1.5)
        if use_highk_hifi:
            request_kmax = float(max(request_kmax, self.config.camb.camb_hifi_highk_kmax * 1.25))
        pars.set_matter_power(redshifts=[0.0], kmax=request_kmax)
        
        if is_lofi:
            tier = self.config.camb.lofi_speed_tier
            if tier == "L2":
                pars.set_accuracy(AccuracyBoost=0.5, lSampleBoost=0.5, lAccuracyBoost=0.5)
            else:
                pars.set_accuracy(AccuracyBoost=1.5, lSampleBoost=1.5, lAccuracyBoost=1.5)
            if self.config.camb.lofi_strategy == "camb_low_accuracy":
                preset = self.config.camb.lofi_accuracy_preset
                pars.set_accuracy(
                    AccuracyBoost=float(preset.get("AccuracyBoost", 0.5)),
                    lSampleBoost=float(preset.get("lSampleBoost", 0.5)),
                    lAccuracyBoost=float(preset.get("lAccuracyBoost", 0.5)),
                )
        else:
            if use_highk_hifi:
                pars.set_accuracy(
                    AccuracyBoost=float(self.config.camb.camb_hifi_accuracy_boost),
                    lSampleBoost=float(self.config.camb.camb_hifi_sampling_boost),
                    lAccuracyBoost=float(self.config.camb.camb_hifi_l_accuracy_boost),
                )
                try:
                    camb.set_halofit_version(str(self.config.camb.camb_hifi_halofit_version))
                except Exception:
                    pass
                if hasattr(pars, "Transfer"):
                    try:
                        pars.Transfer.high_precision = bool(self.config.camb.camb_hifi_use_high_precision_transfer)
                    except Exception:
                        pass
                    try:
                        pars.Transfer.k_per_logint = int(self.config.camb.camb_hifi_k_per_logint)
                    except Exception:
                        pass
            else:
                pars.set_accuracy(AccuracyBoost=1.5, lSampleBoost=1.5, lAccuracyBoost=1.5)
            
        as_scaled = float(min(float(A) * 1.0e-9, 1.95e-8))
        pars.InitPower.set_params(ns=ns, As=as_scaled)
        
        # 新方案主目标是非线性物质功率谱 P_mm；保留 dark_matter 标签以兼容原脚本入口。
        var_species = 'delta_tot'
        if self.config.camb.spectrum_type == 'galaxy':
            var_species = 'delta_tot'

        pars.NonLinear = camb.model.NonLinear_none
        try:
            res_lin = camb.get_results(pars)
        except Exception as exc:
            raise
        k_req_min = float(np.min(k_bins))
        k_req_max = float(np.max(k_bins))
        camb_minkh = max(K_RANGE_MIN, k_req_min * 0.8)
        camb_maxkh = min(K_RANGE_MAX * 1.2, k_req_max * 1.2)
        spectrum_npoints = max(512, int(len(k_bins)))
        k_lin, _, pk_lin = res_lin.get_matter_power_spectrum(
            minkh=camb_minkh,
            maxkh=camb_maxkh,
            npoints=spectrum_npoints,
            var1=var_species,
            var2=var_species
        )
        needs_extrap_lin = bool(np.min(k_bins) < np.min(k_lin) or np.max(k_bins) > np.max(k_lin))
        if needs_extrap_lin:
            raise ValueError(
                "CAMB linear spectrum does not cover requested k range "
                f"[{k_req_min:.4e}, {k_req_max:.4e}] with returned range "
                f"[{float(np.min(k_lin)):.4e}, {float(np.max(k_lin)):.4e}]."
            )
        p_lin_interp = interp1d(k_lin, pk_lin[0], kind="cubic", bounds_error=True)(k_bins)
        
        if only_linear:
            return p_lin_interp, None
            
        pars.NonLinear = camb.model.NonLinear_both
        res_nl = camb.get_results(pars)
        k_nl, _, pk_nl = res_nl.get_matter_power_spectrum(
            minkh=camb_minkh,
            maxkh=camb_maxkh,
            npoints=spectrum_npoints,
            var1=var_species,
            var2=var_species
        )
        needs_extrap_nl = bool(np.min(k_bins) < np.min(k_nl) or np.max(k_bins) > np.max(k_nl))
        if needs_extrap_nl:
            raise ValueError(
                "CAMB nonlinear spectrum does not cover requested k range "
                f"[{k_req_min:.4e}, {k_req_max:.4e}] with returned range "
                f"[{float(np.min(k_nl)):.4e}, {float(np.max(k_nl)):.4e}]."
            )
        p_nl_interp = interp1d(k_nl, pk_nl[0], kind="cubic", bounds_error=True)(k_bins)
        
        return p_lin_interp, p_nl_interp

    def _f_nu(self, theta_batch: np.ndarray) -> np.ndarray:
        omega_cb = np.maximum(theta_batch[:, 1], 1.0e-12)
        h = np.maximum(theta_batch[:, 2] / 100.0, 1.0e-12)
        sum_m_nu = np.maximum(theta_batch[:, 7], 0.0)
        omega_nu = sum_m_nu / np.maximum(93.14 * (h**2), 1.0e-12)
        omega_m_total = np.maximum(omega_cb + omega_nu, 1.0e-12)
        return omega_nu / omega_m_total

    def _compute_l3_features(
        self,
        theta_batch: np.ndarray,
        anchor_theta: np.ndarray,
        *,
        eta_w: float,
    ) -> dict[str, np.ndarray]:
        theta_anchor_2d = anchor_theta.reshape(1, -1)
        # The validation pipeline parameterizes primordial amplitude with A_s,
        # so the L3 feature keeps the original sigma8 slot name but uses A_s as a proxy.
        sigma8_a = float(max(theta_anchor_2d[0, 4], 1.0e-12))
        ns_a = float(theta_anchor_2d[0, 3])
        h_a = float(max(theta_anchor_2d[0, 2] / 100.0, 1.0e-12))
        omega_cb_a = float(max(theta_anchor_2d[0, 1], 1.0e-12))
        omega_b_a = float(max(theta_anchor_2d[0, 0], 1.0e-12))
        omh_a = float(max(omega_cb_a * (h_a**2), 1.0e-12))
        fb_a = float(omega_b_a / omega_cb_a)
        fnu_a = float(self._f_nu(theta_anchor_2d)[0])
        om_a = omega_cb_a
        w0_a = float(theta_anchor_2d[0, 5])
        wa_a = float(theta_anchor_2d[0, 6])
        sigma8 = np.maximum(theta_batch[:, 4], 1.0e-12)
        omega_cb = np.maximum(theta_batch[:, 1], 1.0e-12)
        omega_b = np.maximum(theta_batch[:, 0], 1.0e-12)
        h = np.maximum(theta_batch[:, 2] / 100.0, 1.0e-12)
        delta_log_sigma8 = np.log(sigma8 / sigma8_a)
        delta_ns = theta_batch[:, 3] - ns_a
        delta_log_keq = np.log(
            np.maximum(omega_cb * (h**2), 1.0e-12) / omh_a
        )
        delta_fb = (omega_b / omega_cb) - fb_a
        delta_fnu = self._f_nu(theta_batch) - fnu_a
        delta_weff = (theta_batch[:, 5] - w0_a) + float(eta_w) * (theta_batch[:, 6] - wa_a)
        delta_growth_m = np.log(omega_cb / om_a)
        delta_h = np.log(h / h_a)
        return {
            "DeltaLogSigma8": delta_log_sigma8,
            "DeltaNs": delta_ns,
            "DeltaLogKeq": delta_log_keq,
            "DeltaFb": delta_fb,
            "DeltaFnu": delta_fnu,
            "DeltaWeff": delta_weff,
            "DeltaGrowthM": delta_growth_m,
            "DeltaH": delta_h,
        }

    def _build_anchor_derivative_basis(
        self,
        k_bins: np.ndarray,
        anchor_linear: np.ndarray,
        *,
        k_pivot: float,
    ) -> dict[str, np.ndarray]:
        logk = np.log(np.maximum(k_bins, 1.0e-30))
        log_anchor = np.log(np.maximum(anchor_linear, self.config.eps_r))
        g1 = np.gradient(log_anchor, logk)
        g2 = np.gradient(g1, logk)
        x = np.log(np.maximum(k_bins, 1.0e-30) / max(float(k_pivot), 1.0e-12))
        w_eq = np.exp(-0.5 * ((np.log10(np.maximum(k_bins, 1.0e-30)) + 0.7) / 0.45) ** 2)
        return {"D1": x, "D2": g1, "D3": g2, "D4": w_eq * g1}

    def _default_physics_basis(self, k_bins: np.ndarray) -> dict[str, np.ndarray]:
        logk = np.log10(np.maximum(k_bins, 1.0e-30))
        f1 = np.exp(-0.5 * ((logk + 0.35) / 0.45) ** 2)
        f2 = np.exp(-0.5 * ((logk - 0.15) / 0.55) ** 2)
        f3 = 1.0 / (1.0 + np.exp(-(logk + 0.1) / 0.25))
        f4 = 1.0 / (1.0 + np.exp(-(logk - 0.45) / 0.25))
        f5 = np.exp(-0.5 * ((logk + 1.2) / 0.55) ** 2)
        f6 = np.exp(-0.5 * ((logk + 0.35) / 0.70) ** 2)
        r1 = np.exp(-0.5 * ((logk + 0.2) / 0.6) ** 2)
        r2 = np.exp(-0.5 * ((logk - 0.15) / 0.7) ** 2)
        r3 = 1.0 / (1.0 + np.exp(-(logk - 0.25) / 0.3))
        r4 = np.exp(-0.5 * ((logk + 0.6) / 0.8) ** 2)
        return {
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "F4": f4,
            "F5": f5,
            "F6": f6,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "R4": r4,
        }

    def _resolve_lofi_formula_assets(self, k_bins: np.ndarray) -> dict[str, object]:
        defaults = {
            "formula_name": str(self.config.camb.lofi_formula_name),
            "k_pivot": float(self.config.camb.lofi_formula_pivot_k),
            "eta_w": float(self.config.camb.lofi_formula_eta_w),
            "ratio_mode": "freeze_anchor" if bool(self.config.camb.lofi_formula_freeze_ratio) else "delta_log_r",
            "coefficients": {
                "alpha_A": 1.0,
                "alpha_ns": 1.0,
                "alpha_eq1": 0.45,
                "alpha_eq2": 0.30,
                "alpha_eq3": 0.18,
                "alpha_b1": 0.35,
                "alpha_b2": 0.20,
                "alpha_nu1": 0.45,
                "alpha_nu2": 0.30,
                "alpha_de1": 0.25,
                "alpha_de2": 0.15,
                "beta_beq": 0.20,
                "beta_nus8": 0.20,
                "c_Omega": 0.25,
                "c_h": 0.10,
                "c_de": 0.20,
                "c_nuA": 0.20,
            },
            "ratio_coefficients": {
                "rho_sigma": 0.08,
                "rho_m": 0.06,
                "rho_nu": 0.08,
                "rho_de": 0.05,
            },
            "basis_curves": {},
            "ratio_basis_curves": {},
        }
        asset_path = Path(str(self.config.camb.lofi_formula_asset_path))
        if not asset_path.is_absolute():
            asset_path = Path(self.config.project_root) / asset_path
        if not asset_path.exists():
            return defaults
        try:
            payload = json.loads(asset_path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        merged = dict(defaults)
        merged.update({k: v for k, v in payload.items() if k in {"formula_name", "k_pivot", "eta_w", "ratio_mode"}})
        merged["coefficients"] = {
            **defaults["coefficients"],
            **{str(k): float(v) for k, v in dict(payload.get("coefficients", {})).items()},
        }
        merged["ratio_coefficients"] = {
            **defaults["ratio_coefficients"],
            **{str(k): float(v) for k, v in dict(payload.get("ratio_coefficients", {})).items()},
        }
        k_payload = np.asarray(payload.get("k_bins", []), dtype=np.float64)
        logk = np.log10(np.maximum(k_bins, 1.0e-30))
        if k_payload.ndim == 1 and k_payload.shape[0] >= 2:
            logk_payload = np.log10(np.maximum(k_payload, 1.0e-30))
            basis_curves: dict[str, np.ndarray] = {}
            for key, curve in dict(payload.get("basis_curves", {})).items():
                c = np.asarray(curve, dtype=np.float64)
                if c.ndim == 1 and c.shape[0] == k_payload.shape[0]:
                    basis_curves[str(key)] = np.interp(logk, logk_payload, c).astype(np.float64)
            merged["basis_curves"] = basis_curves
            ratio_basis_curves: dict[str, np.ndarray] = {}
            for key, curve in dict(payload.get("ratio_basis_curves", {})).items():
                c = np.asarray(curve, dtype=np.float64)
                if c.ndim == 1 and c.shape[0] == k_payload.shape[0]:
                    ratio_basis_curves[str(key)] = np.interp(logk, logk_payload, c).astype(np.float64)
            merged["ratio_basis_curves"] = ratio_basis_curves
        return merged

    def _run_l3_formula_batch(
        self,
        theta_batch: np.ndarray,
        k_bins: np.ndarray,
        *,
        anchor_theta: np.ndarray,
        anchor_linear: np.ndarray,
        anchor_ratio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        formula_assets = self._resolve_lofi_formula_assets(k_bins)
        coeffs = dict(formula_assets.get("coefficients", {}))
        ratio_coeffs = dict(formula_assets.get("ratio_coefficients", {}))
        k_pivot = float(formula_assets.get("k_pivot", self.config.camb.lofi_formula_pivot_k))
        eta_w = float(formula_assets.get("eta_w", self.config.camb.lofi_formula_eta_w))
        formula_name = str(formula_assets.get("formula_name", self.config.camb.lofi_formula_name))
        ratio_mode = str(formula_assets.get("ratio_mode", "freeze_anchor")).strip().lower()
        deriv_basis = self._build_anchor_derivative_basis(
            k_bins,
            anchor_linear,
            k_pivot=k_pivot,
        )
        physics_basis = self._default_physics_basis(k_bins)
        for key, curve in dict(formula_assets.get("basis_curves", {})).items():
            physics_basis[str(key)] = np.asarray(curve, dtype=np.float64)
        for key, curve in dict(formula_assets.get("ratio_basis_curves", {})).items():
            physics_basis[str(key)] = np.asarray(curve, dtype=np.float64)
        features = self._compute_l3_features(theta_batch, anchor_theta, eta_w=eta_w)
        delta_a = (
            2.0 * features["DeltaLogSigma8"]
            + float(coeffs.get("c_Omega", 0.0)) * features["DeltaGrowthM"]
            + float(coeffs.get("c_h", 0.0)) * features["DeltaH"]
            + float(coeffs.get("c_de", 0.0)) * features["DeltaWeff"]
            - float(coeffs.get("c_nuA", 0.0)) * features["DeltaFnu"]
        )
        n = theta_batch.shape[0]
        nk = k_bins.shape[0]
        log_delta = np.zeros((n, nk), dtype=np.float64)
        log_delta += float(coeffs.get("alpha_A", 1.0)) * delta_a[:, None]
        log_delta += float(coeffs.get("alpha_ns", 1.0)) * features["DeltaNs"][:, None] * deriv_basis["D1"][None, :]
        log_delta += float(coeffs.get("alpha_eq1", 0.0)) * features["DeltaLogKeq"][:, None] * deriv_basis["D2"][None, :]
        log_delta += float(coeffs.get("alpha_eq2", 0.0)) * features["DeltaLogKeq"][:, None] * deriv_basis["D4"][None, :]
        log_delta += float(coeffs.get("alpha_eq3", 0.0)) * (features["DeltaLogKeq"][:, None] ** 2) * deriv_basis["D3"][None, :]
        log_delta += float(coeffs.get("alpha_b1", 0.0)) * features["DeltaFb"][:, None] * physics_basis["F1"][None, :]
        log_delta += float(coeffs.get("alpha_b2", 0.0)) * features["DeltaFb"][:, None] * physics_basis["F2"][None, :]
        log_delta += float(coeffs.get("alpha_nu1", 0.0)) * features["DeltaFnu"][:, None] * physics_basis["F3"][None, :]
        log_delta += float(coeffs.get("alpha_nu2", 0.0)) * features["DeltaFnu"][:, None] * physics_basis["F4"][None, :]
        log_delta += float(coeffs.get("alpha_de1", 0.0)) * features["DeltaWeff"][:, None] * physics_basis["F5"][None, :]
        log_delta += float(coeffs.get("alpha_de2", 0.0)) * features["DeltaWeff"][:, None] * physics_basis["F6"][None, :]
        log_delta += float(coeffs.get("beta_beq", 0.0)) * (
            features["DeltaFb"][:, None] * features["DeltaLogKeq"][:, None] * physics_basis["F1"][None, :]
        )
        log_delta += float(coeffs.get("beta_nus8", 0.0)) * (
            features["DeltaFnu"][:, None] * features["DeltaLogSigma8"][:, None] * physics_basis["F4"][None, :]
        )
        clip_value = float(self.config.camb.lofi_formula_clip_log_delta)
        log_delta = np.clip(log_delta, -clip_value, clip_value)
        log_anchor = np.log(np.maximum(anchor_linear, self.config.eps_r))[None, :]
        p_linear_batch = np.exp(log_anchor + log_delta)
        if ratio_mode == "delta_log_r":
            log_r = np.log(np.maximum(anchor_ratio, self.config.eps_r))[None, :]
            delta_log_r = np.zeros((n, nk), dtype=np.float64)
            delta_log_r += float(ratio_coeffs.get("rho_sigma", 0.0)) * features["DeltaLogSigma8"][:, None] * physics_basis["R1"][None, :]
            delta_log_r += float(ratio_coeffs.get("rho_m", 0.0)) * features["DeltaGrowthM"][:, None] * physics_basis["R2"][None, :]
            delta_log_r += float(ratio_coeffs.get("rho_nu", 0.0)) * features["DeltaFnu"][:, None] * physics_basis["R3"][None, :]
            delta_log_r += float(ratio_coeffs.get("rho_de", 0.0)) * features["DeltaWeff"][:, None] * physics_basis["R4"][None, :]
            delta_log_r = np.clip(delta_log_r, -0.25, 0.25)
            ratio_batch = np.exp(log_r + delta_log_r)
        else:
            ratio_mode = "freeze_anchor"
            ratio_batch = np.broadcast_to(
                np.maximum(anchor_ratio, self.config.eps_r)[None, :],
                (n, nk),
            ).copy()
        p_nonlin_batch = np.maximum(p_linear_batch * ratio_batch, self.config.eps_r)
        return (
            np.maximum(p_linear_batch, self.config.eps_r),
            p_nonlin_batch,
            formula_name,
            ratio_mode,
        )

    def get_linear_pk(self, theta, k_bins, accuracy_config):
        theta_array = self._coerce_theta(theta)
        k_array = self._coerce_k_bins(k_bins)
        if not self.config.camb.allow_placeholder_backend:
            p_lin, _ = self._call_real_camb(theta_array, k_array, is_lofi=True, only_linear=True)
            return p_lin
        return self._placeholder_linear_pk(theta_array, k_array, accuracy_config)

    def run_lofi_cloud(
        self,
        thetas,
        k_bins,
        noise_config,
        asset_version,
        anchor_ratio=None,
        anchor_theta: ArrayLike | None = None,
        anchor_linear: ArrayLike | None = None,
        progress_context: dict[str, int | str] | None = None,
    ):
        theta_array = self._coerce_thetas(thetas)
        k_array = self._coerce_k_bins(k_bins)
        p_linear_list = []
        p_nonlin_list = []
        ratio_mode = "direct"
        lofi_formula_name = ""
        
        use_ratio = self.config.camb.lofi_strategy == "linear_ratio_transfer" and anchor_ratio is not None
        ratio_array = None if anchor_ratio is None else np.asarray(anchor_ratio, dtype=np.float64)
        if ratio_array is not None and (ratio_array.ndim != 1 or ratio_array.shape[0] != k_array.shape[0]):
            raise ValueError("anchor_ratio must have shape [N_k] and align with k_bins.")
        tier = self.config.camb.lofi_speed_tier
        l2_backend = str(getattr(self.config.camb, "lofi_l2_backend", "legacy")).strip().lower()
        if not self.config.camb.allow_placeholder_backend and tier == "L2" and l2_backend == "gp_emulator":
            raise RuntimeError(
                "The GP-L2 LoFi backend has been removed from this repository. "
                "Set camb.lofi_l2_backend to 'legacy' to continue."
            )
        # L3: 极速近似，完全跳过 CAMB 线性谱调用
        if (
            not self.config.camb.allow_placeholder_backend
            and use_ratio
            and tier == "L3"
            and anchor_theta is not None
            and anchor_linear is not None
        ):
            anchor_theta_array = self._coerce_theta(anchor_theta)
            anchor_linear_array = np.asarray(anchor_linear, dtype=np.float64)
            if anchor_linear_array.ndim != 1 or anchor_linear_array.shape[0] != k_array.shape[0]:
                raise ValueError("anchor_linear must have shape [N_k] and align with k_bins.")
            self._report_lofi_point_progress(
                1,
                max(theta_array.shape[0], 1),
                "物理近似(L3)",
                progress_context=progress_context,
            )
            p_linear_batch, p_nonlin_lofi, lofi_formula_name, ratio_mode = self._run_l3_formula_batch(
                theta_array,
                k_array,
                anchor_theta=anchor_theta_array,
                anchor_linear=anchor_linear_array,
                anchor_ratio=np.asarray(ratio_array, dtype=np.float64),
            )
            self._report_lofi_point_progress(
                max(theta_array.shape[0], 1),
                max(theta_array.shape[0], 1),
                "物理近似(L3)",
                progress_context=progress_context,
                force_newline=True,
            )
            corrupted_count = 0
            accuracy = CAMBAccuracyConfig(mode="real-L3-formula")
        else:
            method_label = self._lofi_compute_method_label(ratio_transfer_applied=use_ratio, tier=tier)
            if tier == "L3" and not (
                use_ratio and anchor_theta is not None and anchor_linear is not None
            ):
                print(
                    "[LoFi] L3 快路径未命中，已回退 CAMB 计算。"
                    "请检查 strategy/anchor_ratio/anchor_theta/anchor_linear 是否齐备。",
                    file=sys.stderr,
                    flush=True,
                )
            for idx, theta in enumerate(theta_array):
                self._report_lofi_point_progress(
                    idx + 1,
                    theta_array.shape[0],
                    method_label,
                    progress_context=progress_context,
                )
                if not self.config.camb.allow_placeholder_backend:
                    p_lin, p_nl = self._call_real_camb(theta, k_array, is_lofi=True, only_linear=use_ratio)
                else:
                    p_lin = self._placeholder_linear_pk(theta, k_array, CAMBAccuracyConfig(mode="placeholder-lofi"))
                    p_nl = self._placeholder_nonlinear_pk(theta, k_array, CAMBAccuracyConfig(mode="placeholder-lofi"))
                p_linear_list.append(np.maximum(np.asarray(p_lin, dtype=np.float64), self.config.eps_r))
                if use_ratio:
                    p_nonlin_list.append(np.maximum(np.asarray(p_lin, dtype=np.float64) * ratio_array, self.config.eps_r))
                else:
                    p_nonlin_list.append(np.maximum(np.asarray(p_nl, dtype=np.float64), self.config.eps_r))
            p_linear_batch = np.vstack(p_linear_list)
            p_nonlin_lofi = np.vstack(p_nonlin_list)
            if not self.config.camb.allow_placeholder_backend:
                accuracy = CAMBAccuracyConfig(mode=f"real-{tier}")
                corrupted_count = 0
            else:
                accuracy = CAMBAccuracyConfig(mode="placeholder-lofi")
                p_nonlin_lofi = self._apply_controlled_noise(p_nonlin_lofi, noise_config)
                p_nonlin_lofi, corrupted_count = self._apply_controlled_outliers(p_nonlin_lofi, noise_config)
        
        if not self.config.camb.allow_placeholder_backend and self.config.camb.lofi_strategy == "noise":
            p_nonlin_lofi = self._apply_controlled_noise(p_nonlin_lofi, noise_config)
            p_nonlin_lofi, corrupted_count = self._apply_controlled_outliers(p_nonlin_lofi, noise_config)

        if getattr(sys.stdout, "isatty", lambda: False)() or getattr(sys.stderr, "isatty", lambda: False)():
            print(file=sys.stderr, flush=True)

        return {
            "asset_version": asset_version,
            "spectrum_type": self.config.camb.spectrum_type,
            "thetas": theta_array,
            "k_bins": k_array,
            "P_linear_batch": p_linear_batch,
            "P_nonlin_lofi": p_nonlin_lofi,
            "noise_injected": bool(
                noise_config.inject_white_noise
                and (self.config.camb.allow_placeholder_backend or self.config.camb.lofi_strategy == "noise")
            ),
            "outliers_injected": bool(
                noise_config.inject_outliers
                and (self.config.camb.allow_placeholder_backend or self.config.camb.lofi_strategy == "noise")
            ),
            "corrupted_ratio": noise_config.corrupted_ratio,
            "corrupted_count": int(corrupted_count),
            "ratio_transfer_applied": bool(use_ratio),
            "ratio_mode": ratio_mode,
            "lofi_formula_name": lofi_formula_name,
            "lofi_speed_tier": tier,
            "accuracy_config": asdict(accuracy),
            "noise_config": asdict(noise_config),
            "warning": asdict(self._backend_warning()),
        }

    def run_hifi_anchor(self, theta, k_bins, accuracy_config, asset_version):
        theta_array = self._coerce_theta(theta)
        k_array = self._coerce_k_bins(k_bins)
        mode = str(getattr(accuracy_config, "mode", "hifi")).strip().lower()
        if (
            _is_hifi_highk_mode(mode)
            and bool(self.config.camb.camb_hifi_require_real_camb)
            and bool(self.config.camb.allow_placeholder_backend)
        ):
            raise RuntimeError(
                "HiFi backend requires real CAMB high-k enhanced mode; "
                "disable `camb.allow_placeholder_backend` before running."
            )
        if not self.config.camb.allow_placeholder_backend:
            p_lin, p_nl = self._call_real_camb(
                theta_array,
                k_array,
                is_lofi=False,
                accuracy_config=accuracy_config,
            )
        else:
            p_lin = self._placeholder_linear_pk(theta_array, k_array, accuracy_config)
            p_nl = self._placeholder_nonlinear_pk(theta_array, k_array, accuracy_config)
        backend_mode = (
            f"real-{mode}" if not self.config.camb.allow_placeholder_backend else f"placeholder-{mode}"
        )
        return {
            "asset_version": asset_version,
            "spectrum_type": self.config.camb.spectrum_type,
            "P_linear": p_lin,
            "P_nonlin_hifi": p_nl,
            "accuracy_config": asdict(CAMBAccuracyConfig(mode=backend_mode)),
            "warning": asdict(self._backend_warning()),
        }

    def run_hifi_anchors(self, thetas_batch, k_bins, accuracy_config, asset_version):
        theta_array = self._coerce_thetas(thetas_batch)
        k_array = self._coerce_k_bins(k_bins)
        p_linear_list: list[np.ndarray] = []
        p_nonlin_list: list[np.ndarray] = []
        for theta in theta_array:
            res = self.run_hifi_anchor(theta=theta, k_bins=k_array, accuracy_config=accuracy_config, asset_version=asset_version)
            p_linear_list.append(np.asarray(res["P_linear"], dtype=np.float64))
            p_nonlin_list.append(np.asarray(res["P_nonlin_hifi"], dtype=np.float64))
        return {
            "asset_version": asset_version,
            "spectrum_type": self.config.camb.spectrum_type,
            "thetas_batch": theta_array,
            "k_bins": k_array,
            "P_linear_batch": np.vstack(p_linear_list),
            "P_nonlin_hifi_batch": np.vstack(p_nonlin_list),
            "accuracy_config": asdict(
                CAMBAccuracyConfig(
                    mode=(
                        f"real-{str(getattr(accuracy_config, 'mode', 'hifi')).strip().lower()}"
                        if not self.config.camb.allow_placeholder_backend
                        else f"placeholder-{str(getattr(accuracy_config, 'mode', 'hifi')).strip().lower()}"
                    )
                )
            ),
            "warning": asdict(self._backend_warning()),
        }
