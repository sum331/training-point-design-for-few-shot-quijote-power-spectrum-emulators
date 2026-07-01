"""Data-provider adapter backed by an isolated Quijote GP surrogate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from z2quijote.runtime_core.quijote_gp_surrogate import load_quijote_gp_surrogate


class QuijoteCAMBLinearAnchorProvider:
    """Compute CAMB linear P(k) anchors for Quijote's 5D BSQ parameterization."""

    provider_name = "camb_sigma8_calibrated"
    anchor_mode = "linear"
    power_label = "linear"

    def __init__(
        self,
        *,
        reference_as: float = 2.1e-9,
        power_eps: float = 1.0e-12,
        min_camb_points: int = 512,
    ) -> None:
        self.reference_as = float(reference_as)
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.min_camb_points = int(max(16, min_camb_points))

    @staticmethod
    def quijote_theta_to_camb_theta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape != (5,):
            raise ValueError(f"Quijote theta must have shape [5], got {theta_arr.shape}.")
        omega_m, omega_b, h, n_s, sigma_8 = (float(value) for value in theta_arr)
        if omega_m <= omega_b:
            raise ValueError(
                "Quijote theta is invalid for CAMB linear anchor: Omega_m must exceed Omega_b, "
                f"got Omega_m={omega_m:.6g}, Omega_b={omega_b:.6g}."
            )
        if h <= 0.0 or sigma_8 <= 0.0:
            raise ValueError(
                "Quijote theta is invalid for CAMB linear anchor: h and sigma_8 must be positive."
            )
        return np.asarray(
            [
                omega_b,
                omega_m,
                100.0 * h,
                n_s,
                sigma_8,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def _build_camb_params(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
    ) -> Any:
        import camb

        omega_m, omega_b, h, n_s, _ = (float(value) for value in theta)
        omega_cdm = omega_m - omega_b
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=100.0 * h,
            ombh2=omega_b * h * h,
            omch2=omega_cdm * h * h,
            mnu=0.0,
        )
        pars.InitPower.set_params(ns=n_s, As=float(primordial_as))
        request_kmax = float(max(np.max(k_bins) * 1.25, 0.1))
        pars.set_matter_power(redshifts=[0.0], kmax=request_kmax)
        pars.NonLinear = camb.model.NonLinear_none
        return pars

    def _reference_sigma8_and_linear_pk(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
    ) -> tuple[float, np.ndarray]:
        import camb

        pars = self._build_camb_params(theta, k_bins, primordial_as=primordial_as)
        results = camb.get_results(pars)
        sigma8_values = np.asarray(results.get_sigma8(), dtype=np.float64).reshape(-1)
        if sigma8_values.size <= 0 or not np.isfinite(sigma8_values[0]) or sigma8_values[0] <= 0.0:
            raise RuntimeError("CAMB returned an invalid sigma8 during Quijote anchor calibration.")
        k_min = float(np.min(k_bins))
        k_max = float(np.max(k_bins))
        camb_minkh = max(k_min * 0.8, 1.0e-5)
        camb_maxkh = max(k_max * 1.2, k_max + 1.0e-6)
        npoints = int(max(self.min_camb_points, k_bins.shape[0]))
        k_camb, _, pk_camb = results.get_matter_power_spectrum(
            minkh=camb_minkh,
            maxkh=camb_maxkh,
            npoints=npoints,
            var1="delta_tot",
            var2="delta_tot",
        )
        if float(np.min(k_bins)) < float(np.min(k_camb)) or float(np.max(k_bins)) > float(np.max(k_camb)):
            raise ValueError(
                "CAMB linear anchor does not cover requested Quijote k range "
                f"[{k_min:.4e}, {k_max:.4e}] with returned range "
                f"[{float(np.min(k_camb)):.4e}, {float(np.max(k_camb)):.4e}]."
        )
        linear = interp1d(k_camb, pk_camb[0], kind="cubic", bounds_error=True)(k_bins)
        return float(sigma8_values[0]), np.maximum(np.asarray(linear, dtype=np.float64), self.power_eps)

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape != (5,):
            raise ValueError(f"Quijote theta must have shape [5], got {theta_arr.shape}.")
        k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_arr.ndim != 1 or k_arr.size <= 0 or np.any(k_arr <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        target_sigma8 = float(theta_arr[4])
        reference_sigma8, reference_linear = self._reference_sigma8_and_linear_pk(
            theta_arr,
            k_arr,
            primordial_as=self.reference_as,
        )
        scale = (target_sigma8 / reference_sigma8) ** 2
        return np.maximum(reference_linear * scale, self.power_eps)


class QuijoteCAMBHMCODE2020AnchorProvider:
    """Compute matter-only CAMB/HMCODE2020 anchors on Quijote's fixed 5D slice."""

    provider_name = "camb_hmcode2020_sigma8_calibrated"
    anchor_mode = "hmcode2020"
    power_label = "hmcode2020"
    matter_power_var = "delta_tot"

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

    @staticmethod
    def _coerce_theta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape != (5,):
            raise ValueError(f"Quijote theta must have shape [5], got {theta_arr.shape}.")
        omega_m, omega_b, h, _, sigma_8 = (float(value) for value in theta_arr)
        if omega_m <= omega_b:
            raise ValueError(
                "Quijote theta is invalid for CAMB HMCODE2020 anchor: Omega_m must exceed "
                f"Omega_b, got Omega_m={omega_m:.6g}, Omega_b={omega_b:.6g}."
            )
        if h <= 0.0 or sigma_8 <= 0.0:
            raise ValueError(
                "Quijote theta is invalid for CAMB HMCODE2020 anchor: h and sigma_8 must be positive."
            )
        return theta_arr.astype(np.float64)

    def _coerce_k_bins(self, k_bins: Sequence[float] | np.ndarray) -> np.ndarray:
        k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_arr.ndim != 1 or k_arr.size <= 0 or np.any(k_arr <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        return k_arr

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

    def _sigma8_for_as(
        self,
        theta: np.ndarray,
        k_bins: np.ndarray,
        *,
        primordial_as: float,
    ) -> float:
        import camb

        pars = self._build_camb_params(
            theta,
            k_bins,
            primordial_as=primordial_as,
            nonlinear=False,
        )
        results = camb.get_results(pars)
        sigma8_values = np.asarray(results.get_sigma8(), dtype=np.float64).reshape(-1)
        if sigma8_values.size <= 0 or not np.isfinite(sigma8_values[0]) or sigma8_values[0] <= 0.0:
            raise RuntimeError("CAMB returned an invalid sigma8 during HMCODE2020 anchor calibration.")
        return float(sigma8_values[0])

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
                "CAMB HMCODE2020 anchor does not cover requested Quijote k range "
                f"[{k_min:.4e}, {k_max:.4e}] with returned range "
                f"[{float(np.min(k_camb)):.4e}, {float(np.max(k_camb)):.4e}]."
            )
        hmcode = interp1d(k_camb, pk_camb[0], kind="cubic", bounds_error=True)(k_bins)
        return np.maximum(np.asarray(hmcode, dtype=np.float64), self.power_eps)

    def get_anchor_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        theta_arr = self._coerce_theta(theta)
        k_arr = self._coerce_k_bins(k_bins)
        target_sigma8 = float(theta_arr[4])
        reference_sigma8 = self._sigma8_for_as(
            theta_arr,
            k_arr,
            primordial_as=self.reference_as,
        )
        target_as = self.reference_as * (target_sigma8 / reference_sigma8) ** 2
        return self._hmcode_pk_for_as(theta_arr, k_arr, primordial_as=target_as)

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Compatibility shim for the legacy p_linear_batch anchor slot."""

        return self.get_anchor_pk(theta, k_bins)


class QuijoteCAMBCDMMHMCODE2020AnchorProvider(QuijoteCAMBHMCODE2020AnchorProvider):
    """Compute CAMB/HMCODE2020 CDM auto-power anchors on Quijote's fixed 5D slice."""

    provider_name = "camb_cdm_hmcode2020_sigma8_calibrated"
    power_label = "cdm_hmcode2020"
    matter_power_var = "delta_cdm"


class QuijoteOfficialLinearAnchorProvider:
    """Read official Quijote linear P(k) tables for exact BSQ source points."""

    provider_name = "quijote_official_linear_pk"

    def __init__(
        self,
        *,
        raw_root: str | Path,
        params_file: str | Path,
        file_name: str = "Pk_mm_z=0.000.txt",
        normfac_file_name: str = "Normfac.txt",
        apply_normfac: bool = True,
        power_eps: float = 1.0e-12,
        theta_match_atol: float = 1.0e-8,
        theta_match_rtol: float = 1.0e-8,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.params_file = Path(params_file)
        self.file_name = str(file_name)
        self.normfac_file_name = str(normfac_file_name)
        self.apply_normfac = bool(apply_normfac)
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.theta_match_atol = float(max(theta_match_atol, 0.0))
        self.theta_match_rtol = float(max(theta_match_rtol, 0.0))
        self.raw_thetas = self._load_thetas()
        self._table_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _load_thetas(self) -> np.ndarray:
        if not self.params_file.exists():
            raise FileNotFoundError(f"Quijote official params file not found: {self.params_file}")
        arr = np.loadtxt(self.params_file, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(
                "Quijote official params file must have shape [n, 5], "
                f"got {arr.shape}."
            )
        return np.asarray(arr, dtype=np.float64)

    def _simulation_index_for_theta(self, theta: Sequence[float] | np.ndarray) -> int:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape != (5,):
            raise ValueError(f"Quijote theta must have shape [5], got {theta_arr.shape}.")
        close = np.all(
            np.isclose(
                self.raw_thetas,
                theta_arr[None, :],
                rtol=self.theta_match_rtol,
                atol=self.theta_match_atol,
            ),
            axis=1,
        )
        matches = np.flatnonzero(close)
        if matches.size:
            return int(matches[0])
        distances = np.max(np.abs(self.raw_thetas - theta_arr[None, :]), axis=1)
        nearest = int(np.argmin(distances))
        raise ValueError(
            "Quijote official linear P(k) is only available for the discrete "
            "32768 BSQ source thetas. The requested theta is not an exact source-bank "
            f"match within atol={self.theta_match_atol:g}, rtol={self.theta_match_rtol:g}. "
            f"Nearest simulation_id={nearest} has max_abs_delta={float(distances[nearest]):.6g}. "
            "Use a continuous Quijote-linear surrogate before requesting arbitrary theta points."
        )

    def _load_linear_table(self, simulation_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._table_cache.get(int(simulation_index))
        if cached is not None:
            return cached
        path = self.raw_root / str(int(simulation_index)) / self.file_name
        if not path.exists():
            raise FileNotFoundError(f"Quijote official linear P(k) file not found: {path}")
        table = np.loadtxt(path, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(f"Quijote official linear table must have at least two columns: {path}")
        k_source = np.asarray(table[:, 0], dtype=np.float64)
        pk_source = np.asarray(table[:, 1], dtype=np.float64)
        if self.apply_normfac:
            normfac_path = self.raw_root / str(int(simulation_index)) / self.normfac_file_name
            if not normfac_path.exists():
                raise FileNotFoundError(f"Quijote official Normfac file not found: {normfac_path}")
            normfac_value = np.loadtxt(normfac_path, dtype=np.float64)
            normfac = float(np.asarray(normfac_value, dtype=np.float64).reshape(-1)[0])
            if not np.isfinite(normfac) or normfac <= 0.0:
                raise ValueError(f"Quijote official Normfac must be positive and finite: {normfac_path}")
            pk_source = pk_source * normfac
        if k_source.size < 2 or np.any(k_source <= 0.0) or not np.all(np.diff(k_source) > 0.0):
            raise ValueError(f"Quijote official linear k grid must be positive and increasing: {path}")
        if np.any(pk_source <= 0.0):
            raise ValueError(f"Quijote official linear P(k) must be positive for log interpolation: {path}")
        cached = (k_source, pk_source)
        self._table_cache[int(simulation_index)] = cached
        return cached

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        k_target = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_target.ndim != 1 or k_target.size <= 0 or np.any(k_target <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        simulation_index = self._simulation_index_for_theta(theta)
        k_source, pk_source = self._load_linear_table(simulation_index)
        k_min = float(np.min(k_target))
        k_max = float(np.max(k_target))
        if k_min < float(k_source[0]) or k_max > float(k_source[-1]):
            raise ValueError(
                "Quijote official linear P(k) table does not cover requested k range "
                f"[{k_min:.6g}, {k_max:.6g}] with source range "
                f"[{float(k_source[0]):.6g}, {float(k_source[-1]):.6g}]."
            )
        interpolator = PchipInterpolator(
            np.log(k_source),
            np.log(np.maximum(pk_source, self.power_eps)),
            extrapolate=False,
        )
        aligned = np.exp(interpolator(np.log(k_target)))
        if np.any(~np.isfinite(aligned)):
            raise RuntimeError("Quijote official linear interpolation produced non-finite values.")
        return np.maximum(np.asarray(aligned, dtype=np.float64), self.power_eps)


class QuijoteLinearGeneratorProvider:
    """Continuous Quijote official-linear P(k) generator used as logdiff anchor."""

    provider_name = "quijote_official_linear_svgp_generator"

    def __init__(
        self,
        *,
        generator: Any | None = None,
        generator_path: str | Path | None = None,
        power_eps: float = 1.0e-12,
        device: str = "cpu",
    ) -> None:
        if generator is None and generator_path is None:
            raise ValueError("Either generator or generator_path must be provided.")
        self.generator = generator if generator is not None else load_quijote_gp_surrogate(generator_path)
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.device = str(device)

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        k_target = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_target.ndim != 1 or k_target.size <= 0 or np.any(k_target <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        kwargs: dict[str, Any] = {
            "input_space": "raw",
            "k_target": k_target,
        }
        if "svgp" in str(self.generator.metadata.get("surrogate_kind", "")).lower():
            kwargs["device"] = self.device
        prediction = self.generator.predict(
            np.asarray(theta, dtype=np.float64).reshape(1, -1),
            **kwargs,
        )
        linear = np.asarray(prediction["pk_mean"][0], dtype=np.float64).reshape(-1)
        if linear.shape != k_target.shape:
            raise RuntimeError(
                "Quijote linear generator returned shape "
                f"{linear.shape}, expected {k_target.shape}."
            )
        if np.any(~np.isfinite(linear)):
            raise RuntimeError("Quijote linear generator returned non-finite values.")
        return np.maximum(linear, self.power_eps)


class QuijoteGPDataProvider:
    """Expose a trained Quijote surrogate through the existing provider-shaped API."""

    def __init__(
        self,
        *,
        surrogate: Any | None = None,
        surrogate_path: str | Path | None = None,
        linear_power_provider: Any | None = None,
        linear_power_reference_as: float = 2.1e-9,
        power_eps: float = 1.0e-12,
        surrogate_device: str = "cpu",
    ) -> None:
        if surrogate is None and surrogate_path is None:
            raise ValueError("Either surrogate or surrogate_path must be provided.")
        self.surrogate = surrogate if surrogate is not None else load_quijote_gp_surrogate(surrogate_path)
        self.linear_power_provider = linear_power_provider or QuijoteCAMBLinearAnchorProvider(
            reference_as=float(linear_power_reference_as),
            power_eps=float(power_eps),
        )
        self.linear_power_provider_name = str(
            getattr(self.linear_power_provider, "provider_name", "camb_sigma8_calibrated")
        )
        self.surrogate_device = str(surrogate_device)

    def _surrogate_predict(
        self,
        theta_batch: Sequence[Sequence[float]] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        kwargs: dict[str, Any] = {
            "input_space": "raw",
            "k_target": np.asarray(k_bins, dtype=np.float64).reshape(-1),
        }
        if "svgp" in str(self.surrogate.metadata.get("surrogate_kind", "")).lower():
            kwargs["device"] = self.surrogate_device
        return self.surrogate.predict(
            np.asarray(theta_batch, dtype=np.float64),
            **kwargs,
        )

    def _linear_anchor(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        provider = self.linear_power_provider
        if hasattr(provider, "get_anchor_pk"):
            return np.asarray(provider.get_anchor_pk(theta, k_bins), dtype=np.float64).reshape(-1)
        if hasattr(provider, "get_linear_pk"):
            return np.asarray(provider.get_linear_pk(theta, k_bins), dtype=np.float64).reshape(-1)
        raise TypeError("linear_power_provider must define get_anchor_pk(theta, k_bins) or get_linear_pk(theta, k_bins).")

    def run_hifi_anchor(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "quijote_gp",
    ) -> dict[str, np.ndarray | dict[str, Any] | None]:
        del accuracy_config
        prediction = self._surrogate_predict(
            np.asarray(theta, dtype=np.float64).reshape(1, -1),
            k_bins,
        )
        linear_anchor = self._linear_anchor(prediction["raw_thetas"][0], prediction["k_bins"])
        return {
            "theta": prediction["raw_thetas"][0].astype(np.float64),
            "k_bins": prediction["k_bins"].astype(np.float64),
            "P_linear": linear_anchor.astype(np.float64),
            "P_nonlin_hifi": prediction["pk_mean"][0].astype(np.float64),
            "metadata": {
                "provider": "quijote_gp",
                "asset_version": str(asset_version),
                "surrogate_target_transform": str(
                    self.surrogate.metadata.get("target_transform", "direct_logpk")
                ),
                "surrogate_kind": str(self.surrogate.metadata.get("surrogate_kind", "unknown")),
                "surrogate_role": str(self.surrogate.metadata.get("role", "source_surrogate")),
                "surrogate_device": self.surrogate_device,
                "linear_anchor_provider": self.linear_power_provider_name,
                "anchor_power_provider": self.linear_power_provider_name,
                "anchor_power_label": str(getattr(self.linear_power_provider, "power_label", "linear")),
                "parameter_space": str(
                    self.surrogate.metadata.get("parameter_space", "quijote_bsq5")
                ),
            },
        }

    def run_hifi_anchors(
        self,
        theta_batch: Sequence[Sequence[float]] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "quijote_gp",
    ) -> list[dict[str, np.ndarray | dict[str, Any] | None]]:
        del accuracy_config
        prediction = self._surrogate_predict(theta_batch, k_bins)
        rows: list[dict[str, np.ndarray | dict[str, Any] | None]] = []
        for row_index in range(prediction["raw_thetas"].shape[0]):
            linear_anchor = self._linear_anchor(
                prediction["raw_thetas"][row_index],
                prediction["k_bins"],
            )
            rows.append(
                {
                    "theta": prediction["raw_thetas"][row_index].astype(np.float64),
                    "k_bins": prediction["k_bins"].astype(np.float64),
                    "P_linear": linear_anchor.astype(np.float64),
                    "P_nonlin_hifi": prediction["pk_mean"][row_index].astype(np.float64),
                    "metadata": {
                        "provider": "quijote_gp",
                        "asset_version": str(asset_version),
                        "surrogate_target_transform": str(
                            self.surrogate.metadata.get("target_transform", "direct_logpk")
                        ),
                        "surrogate_kind": str(
                            self.surrogate.metadata.get("surrogate_kind", "unknown")
                        ),
                        "surrogate_role": str(
                            self.surrogate.metadata.get("role", "source_surrogate")
                        ),
                        "surrogate_device": self.surrogate_device,
                        "linear_anchor_provider": self.linear_power_provider_name,
                        "anchor_power_provider": self.linear_power_provider_name,
                        "anchor_power_label": str(getattr(self.linear_power_provider, "power_label", "linear")),
                        "parameter_space": str(
                            self.surrogate.metadata.get("parameter_space", "quijote_bsq5")
                        ),
                    },
                }
            )
        return rows


__all__ = [
    "QuijoteCAMBHMCODE2020AnchorProvider",
    "QuijoteCAMBCDMMHMCODE2020AnchorProvider",
    "QuijoteCAMBLinearAnchorProvider",
    "QuijoteGPDataProvider",
    "QuijoteLinearGeneratorProvider",
    "QuijoteOfficialLinearAnchorProvider",
]
