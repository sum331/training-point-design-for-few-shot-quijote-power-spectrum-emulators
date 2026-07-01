"""CSST official-emulator provider for the z2 8D fastmock workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.interpolate import interp1d


CSST_THETA_NAMES: tuple[str, ...] = (
    "Omegab",
    "Omegam",
    "H0",
    "ns",
    "A",
    "w",
    "wa",
    "mnu",
)

DEFAULT_CSST_THETA_BOUNDS: dict[str, tuple[float, float]] = {
    "Omegab": (0.04, 0.06),
    "Omegam": (0.24, 0.40),
    "H0": (60.0, 80.0),
    "ns": (0.92, 1.00),
    "A": (1.7, 2.5),
    "w": (-1.3, -0.7),
    "wa": (-0.5, 0.5),
    "mnu": (0.0, 0.3),
}


def _default_vendor_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "vendor" / "csstemu_official_full"


DEFAULT_VENDOR_PATH = _default_vendor_path()
_CUDA_TRUTH_LOCK = threading.RLock()


def csst_theta_bounds_as_array(
    raw_bounds: Mapping[str, Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(raw_bounds, np.ndarray):
        bounds = np.asarray(raw_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape != (len(CSST_THETA_NAMES), 2):
            raise ValueError(
                "CSST theta_bounds array must have shape "
                f"({len(CSST_THETA_NAMES)}, 2), got {bounds.shape}."
            )
        if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(f"CSST theta_bounds array contains invalid bounds: {bounds!r}.")
        return bounds

    resolved = dict(DEFAULT_CSST_THETA_BOUNDS)
    if raw_bounds is not None:
        for raw_name, raw_pair in raw_bounds.items():
            name = str(raw_name).strip()
            if name not in DEFAULT_CSST_THETA_BOUNDS:
                raise ValueError(f"Unknown CSST theta bound name: {raw_name!r}")
            if not isinstance(raw_pair, Sequence) or len(raw_pair) != 2:
                raise ValueError(f"CSST theta bounds for {name!r} must be a pair.")
            low = float(raw_pair[0])
            high = float(raw_pair[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(f"Invalid CSST theta bounds for {name!r}: {(low, high)!r}")
            resolved[name] = (low, high)

    return np.asarray([resolved[name] for name in CSST_THETA_NAMES], dtype=np.float64)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    resolved = str(path.resolve())
    inserted = resolved not in sys.path
    if inserted:
        sys.path.insert(0, resolved)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(resolved)
            except ValueError:
                pass


def _patch_scipy_simps_compatibility() -> None:
    """Support CSST code paths that still import scipy.integrate.simps."""

    try:
        import scipy.integrate as integrate
    except Exception:
        return
    if not hasattr(integrate, "simps") and hasattr(integrate, "simpson"):
        integrate.simps = integrate.simpson  # type: ignore[attr-defined]


def _coerce_csst_theta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if arr.shape != (len(CSST_THETA_NAMES),):
        raise ValueError(
            "CSST theta must be 8D in order "
            "[Omegab, Omegam, H0, ns, A, w, wa, mnu], "
            f"got shape {arr.shape}."
        )
    if arr[1] <= arr[0]:
        raise ValueError("CSST theta requires Omegam > Omegab so Omegac is positive.")
    return arr.astype(np.float64)


def _coerce_csst_theta_batch(theta: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(CSST_THETA_NAMES):
        raise ValueError(
            "CSST theta batch must have shape "
            f"[N,{len(CSST_THETA_NAMES)}], got {arr.shape}."
        )
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError("CSST theta requires Omegam > Omegab so Omegac is positive.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("CSST theta batch contains non-finite values.")
    return arr.astype(np.float64)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "cublas" in text or "allocation" in text)


class _TorchOfficialResidualRatioEngine:
    """Torch/CUDA port of the official CSST residual-ratio GP path.

    The official emulator predicts PCA coefficients with 20 independent GP
    models and then reconstructs the ratio grid.  This class mirrors that
    calculation for large point clouds; scalar and refinement paths continue to
    use the original CPU implementation.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: Any,
        dtype: Any,
        power_eps: float,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.power_eps = float(max(power_eps, 1.0e-30))
        self.emunamestr = str(getattr(model, "emunamestr", ""))
        self.z_count = int(len(model.zlists))

        self.klist = torch.as_tensor(
            np.asarray(model.klist, dtype=np.float64),
            device=device,
            dtype=dtype,
        )
        self.pca_components = torch.as_tensor(
            np.asarray(model._PCA_components, dtype=np.float64),
            device=device,
            dtype=dtype,
        )
        self.pca_mean = torch.as_tensor(
            np.asarray(model._PCA_mean, dtype=np.float64).reshape(1, -1),
            device=device,
            dtype=dtype,
        )

        if bool(getattr(model, "NormBeforeGP", False)):
            self.param_mean = torch.as_tensor(
                np.asarray(model.paramSS.mean_, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=dtype,
            )
            self.param_scale = torch.as_tensor(
                np.asarray(model.paramSS.scale_, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=dtype,
            )
            self.coeff_mean = torch.as_tensor(
                np.asarray(model.coeffSS.mean_, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=dtype,
            )
            self.coeff_scale = torch.as_tensor(
                np.asarray(model.coeffSS.scale_, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=dtype,
            )
        else:
            nvec = int(len(model._GPR))
            x_dim = int(np.asarray(model.X_train, dtype=np.float64).shape[1])
            self.param_mean = torch.zeros((1, x_dim), device=device, dtype=dtype)
            self.param_scale = torch.ones((1, x_dim), device=device, dtype=dtype)
            self.coeff_mean = torch.zeros((1, nvec), device=device, dtype=dtype)
            self.coeff_scale = torch.ones((1, nvec), device=device, dtype=dtype)

        self.gp_state: list[tuple[Any, Any, Any, Any, Any]] = []
        for gp in model._GPR:
            kernel = gp.kernel
            constant = float(kernel.k1.constant_value)
            length_scale = np.asarray(kernel.k2.length_scale, dtype=np.float64).reshape(-1)
            x_train = np.asarray(gp.X_train, dtype=np.float64)
            if length_scale.size == 1:
                length_scale = np.full((x_train.shape[1],), float(length_scale[0]), dtype=np.float64)
            self.gp_state.append(
                (
                    torch.as_tensor(x_train, device=device, dtype=dtype),
                    torch.as_tensor(length_scale.reshape(1, 1, -1), device=device, dtype=dtype),
                    torch.as_tensor(np.asarray(gp.alpha_, dtype=np.float64).reshape(-1), device=device, dtype=dtype),
                    torch.as_tensor(float(gp._y_train_mean), device=device, dtype=dtype),
                    torch.as_tensor(float(gp._y_train_std), device=device, dtype=dtype),
                )
            )
            self.gp_state[-1] = (
                self.gp_state[-1][0],
                self.gp_state[-1][1],
                self.gp_state[-1][2],
                self.gp_state[-1][3],
                self.gp_state[-1][4],
                torch.as_tensor(constant, device=device, dtype=dtype),
            )

    def predict_ratio(self, ncosmo: np.ndarray, k: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            query = torch.as_tensor(np.asarray(ncosmo, dtype=np.float64), device=self.device, dtype=self.dtype)
            query = (query - self.param_mean) / self.param_scale
            coeff_cols = []
            for x_train, length_scale, alpha, y_mean, y_std, constant in self.gp_state:
                diff = (query[:, None, :] - x_train[None, :, :]) / length_scale
                kernel = constant * torch.exp(-0.5 * torch.sum(diff.square(), dim=2))
                coeff_cols.append((kernel @ alpha) * y_std + y_mean)
            coeff = torch.stack(coeff_cols, dim=1)
            coeff = coeff * self.coeff_scale + self.coeff_mean
            reconstructed = coeff @ self.pca_components + self.pca_mean
            if self.emunamestr[:2] == "lg":
                reconstructed = torch.pow(torch.as_tensor(10.0, device=self.device, dtype=self.dtype), reconstructed)
            grid_values = reconstructed.reshape(query.shape[0], self.z_count, self.klist.numel())
            z0_values = torch.flip(grid_values, dims=(1,))[:, 0, :]
            k_target = torch.as_tensor(np.asarray(k, dtype=np.float64), device=self.device, dtype=self.dtype)
            right = torch.searchsorted(self.klist, k_target, right=False).clamp(1, self.klist.numel() - 1)
            left = right - 1
            k_left = self.klist.index_select(0, left)
            k_right = self.klist.index_select(0, right)
            weight = (k_target - k_left) / torch.clamp(k_right - k_left, min=torch.finfo(self.dtype).tiny)
            left_values = z0_values.index_select(1, left)
            right_values = z0_values.index_select(1, right)
            ratio = left_values * (1.0 - weight.reshape(1, -1)) + right_values * weight.reshape(1, -1)
            ratio = torch.clamp(ratio, min=self.power_eps)
            return ratio.detach().cpu().numpy().astype(np.float64)


@dataclass(slots=True)
class CSSTCAMBNonlinearAnchorProvider:
    """Use CAMB/HMCODE2020 nonlinear P(k) as the v3 logdiff anchor."""

    config: Any
    power_eps: float = 1.0e-12
    provider_name: str = "camb_nonlinear_hmcode2020"
    power_label: str = "hmcode2020"

    def get_anchor_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        from z2quijote.runtime_core.camb_data_provider import CAMBAccuracyConfig, CAMBDataProvider

        theta_arr = _coerce_csst_theta(theta)
        k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        provider = CAMBDataProvider(config=self.config)
        result = provider.run_hifi_anchor(
            theta=theta_arr,
            k_bins=k_arr,
            accuracy_config=CAMBAccuracyConfig(mode="hifi_highk_enhanced"),
            asset_version="csst_camb_hmcode2020_anchor",
        )
        anchor = np.asarray(result["P_nonlin_hifi"], dtype=np.float64).reshape(-1)
        if anchor.shape != k_arr.shape:
            raise RuntimeError(
                "CAMB nonlinear anchor returned shape "
                f"{anchor.shape}, expected {k_arr.shape}."
            )
        return np.maximum(anchor, float(max(self.power_eps, 1.0e-30)))

    def get_linear_pk(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Compatibility shim for the pipeline's legacy p_linear_batch slot."""

        return self.get_anchor_pk(theta, k_bins)


@dataclass(slots=True)
class CSSTDataProvider:
    """Provider-shaped adapter for official CSST z=0 P(k) generation.

    The target spectrum comes from the official CSST Emulator.  The returned
    residual path uses the same internal HMCODE2020 reference that the official
    ``get_pknl(..., nltype='hmcode2020')`` path multiplies by the learned
    nonlinear/HMCODE2020 ratio.
    """

    vendor_path: Path = DEFAULT_VENDOR_PATH
    redshift: float = 0.0
    p_cb: bool = False
    linear_model: str = "Emulator"
    nonlinear_model: str = "hmcode2020"
    neutrino_mass_split: str = "single"
    checkbound: bool = True
    verbose: bool = False
    anchor_provider: Any | None = None
    anchor_provider_name: str = "camb_nonlinear_hmcode2020"
    truth_backend: str = "cpu_batch"
    truth_dtype: str = "float32"
    truth_device: str = "auto"
    power_eps: float = 1.0e-12
    k_min: float = 1.0e-2
    k_max: float = 1.0e1
    _emulator: Any | None = None
    _torch_residual_engine: Any | None = None
    _last_truth_backend_used: str = "cpu_batch"

    def _load_emulator(self) -> Any:
        if self._emulator is not None:
            return self._emulator
        vendor = Path(self.vendor_path).resolve()
        if not vendor.exists():
            raise FileNotFoundError(f"CSST vendor path not found: {vendor}")
        _patch_scipy_simps_compatibility()
        with _prepend_sys_path(vendor):
            from CEmulator.Emulator import Pkmm_CEmulator

        self._emulator = Pkmm_CEmulator(
            verbose=bool(self.verbose),
            neutrino_mass_split=str(self.neutrino_mass_split),
        )
        return self._emulator

    def _coerce_k_bins(self, k_bins: Sequence[float] | np.ndarray) -> np.ndarray:
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k.ndim != 1 or k.size <= 0 or np.any(k <= 0.0):
            raise ValueError("k_bins must be a non-empty positive 1D array.")
        k_min = float(np.min(k))
        k_max = float(np.max(k))
        if k_min < float(self.k_min) or k_max > float(self.k_max):
            raise ValueError(
                "CSST v3 k_bins must stay within "
                f"[{float(self.k_min):.4g}, {float(self.k_max):.4g}] h/Mpc, "
                f"got [{k_min:.4g}, {k_max:.4g}]."
            )
        return k.astype(np.float64)

    def _set_theta(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = _coerce_csst_theta(theta)
        omegab, omegam, h0, ns, amp_1e9_as, w0, wa, mnu = arr
        emulator = self._load_emulator()
        emulator.set_cosmos(
            Omegab=float(omegab),
            Omegac=float(omegam - omegab),
            H0=float(h0),
            As=float(amp_1e9_as) * 1.0e-9,
            ns=float(ns),
            w=float(w0),
            wa=float(wa),
            mnu=float(mnu),
            checkbound=bool(self.checkbound),
        )
        return arr

    def _anchor_pk(self, theta: np.ndarray, k_bins: np.ndarray) -> np.ndarray:
        provider = self.anchor_provider
        if provider is None:
            raise ValueError("CSSTDataProvider requires an anchor_provider for logdiff runs.")
        if hasattr(provider, "get_anchor_pk"):
            anchor = provider.get_anchor_pk(theta, k_bins)
        elif hasattr(provider, "get_linear_pk"):
            anchor = provider.get_linear_pk(theta, k_bins)
        else:
            raise TypeError("anchor_provider must define get_anchor_pk or get_linear_pk.")
        anchor_arr = np.asarray(anchor, dtype=np.float64).reshape(-1)
        if anchor_arr.shape != k_bins.shape:
            raise RuntimeError(
                "CSST anchor provider returned shape "
                f"{anchor_arr.shape}, expected {k_bins.shape}."
            )
        if np.any(~np.isfinite(anchor_arr)):
            raise RuntimeError("CSST anchor provider returned non-finite values.")
        return np.maximum(anchor_arr, float(max(self.power_eps, 1.0e-30)))

    def _official_residual_ratio(self, emulator: Any, z: np.ndarray, k: np.ndarray) -> np.ndarray:
        nltype = str(self.nonlinear_model).strip().lower()
        if not (bool(self.p_cb) or np.isclose(float(emulator.Cosmo.Omeganu), 0.0, atol=1.0e-10)):
            raise ValueError(
                "CSST residual output currently supports Pcb=True or zero-neutrino total matter. "
                "z2 fixes mnu=0, so this should hold for the active 5D slice."
            )
        if nltype == "linear":
            ratio = emulator.Bkcb.get_Bk(z, k)
            anchor_name = "csst_official_linear"
        elif nltype == "halofit":
            ratio = emulator.Bkcb_halofit.get_Bk(z, k)
            anchor_name = "csst_official_halofit"
        elif nltype == "hmcode2020":
            ratio = emulator.Bkcb_hmcode2020.get_Bk(z, k)
            anchor_name = "csst_official_hmcode2020"
        else:
            raise ValueError(f"Unsupported CSST nonlinear_model for residual output: {self.nonlinear_model!r}")
        ratio_arr = np.asarray(ratio, dtype=np.float64).reshape(len(z), len(k))
        if np.any(~np.isfinite(ratio_arr)):
            raise RuntimeError("CSST official residual ratio returned non-finite values.")
        return np.maximum(ratio_arr, float(max(self.power_eps, 1.0e-30))).astype(np.float64)

    def _check_theta_batch_bounds(self, thetas: np.ndarray) -> None:
        if not bool(self.checkbound):
            return
        bounds = csst_theta_bounds_as_array()
        low = bounds[:, 0].reshape(1, -1)
        high = bounds[:, 1].reshape(1, -1)
        if np.any(thetas < low) or np.any(thetas > high):
            bad = np.argwhere((thetas < low) | (thetas > high))[0]
            row_idx, col_idx = int(bad[0]), int(bad[1])
            name = CSST_THETA_NAMES[col_idx]
            value = float(thetas[row_idx, col_idx])
            raise ValueError(
                f"CSST theta batch has out-of-range {name}={value:.6g} at row {row_idx}; "
                f"bounds are [{float(low[0, col_idx]):.6g}, {float(high[0, col_idx]):.6g}]."
            )

    @staticmethod
    def _normalized_theta_batch(thetas: np.ndarray) -> np.ndarray:
        bounds = csst_theta_bounds_as_array()
        low = bounds[:, 0].reshape(1, -1)
        width = (bounds[:, 1] - bounds[:, 0]).reshape(1, -1)
        return (np.asarray(thetas, dtype=np.float64) - low) / width

    @staticmethod
    def _batch_ratio_from_official_gp(model: Any, ncosmo: np.ndarray, k: np.ndarray) -> np.ndarray:
        if bool(getattr(model, "NormBeforeGP", False)):
            norm_cosmo = model.paramSS.transform(ncosmo)
        else:
            norm_cosmo = np.asarray(ncosmo, dtype=np.float64).copy()
        coeff = np.column_stack(
            [np.asarray(gp.predict(norm_cosmo), dtype=np.float64).reshape(-1) for gp in model._GPR]
        )
        if bool(getattr(model, "NormBeforeGP", False)):
            coeff = model.coeffSS.inverse_transform(coeff)
        reconstructed = coeff @ model._PCA_components + model._PCA_mean.reshape(1, -1)
        if str(getattr(model, "emunamestr", ""))[:2] == "lg":
            reconstructed = 10.0**reconstructed
        sample_count = int(ncosmo.shape[0])
        grid_values = reconstructed.reshape(sample_count, len(model.zlists), len(model.klist))
        grid_values = grid_values[:, ::-1, :]
        # z2 uses z=0.0, which is exactly the first row after reversing CSST's
        # descending z grid. CSST's official k interpolation is linear.
        z0_values = grid_values[:, 0, :]
        interpolated = interp1d(
            model.klist,
            z0_values,
            kind="linear",
            axis=1,
            bounds_error=True,
        )(k)
        return np.asarray(interpolated, dtype=np.float64)

    def _official_residual_ratio_batch(self, emulator: Any, thetas: np.ndarray, k: np.ndarray) -> np.ndarray:
        nltype = str(self.nonlinear_model).strip().lower()
        if not bool(self.p_cb) and not np.allclose(thetas[:, 7], 0.0, atol=1.0e-12):
            raise ValueError(
                "CSST residual batch output currently supports Pcb=True or zero-neutrino total matter. "
                "z2 fixes mnu=0 for the active 5D slice."
            )
        if nltype == "linear":
            model = emulator.Bkcb
        elif nltype == "halofit":
            model = emulator.Bkcb_halofit
        elif nltype == "hmcode2020":
            model = emulator.Bkcb_hmcode2020
        else:
            raise ValueError(f"Unsupported CSST nonlinear_model for residual output: {self.nonlinear_model!r}")
        ncosmo = self._normalized_theta_batch(thetas)
        ratio = self._batch_ratio_from_official_gp(model, ncosmo, k)
        if np.any(~np.isfinite(ratio)):
            raise RuntimeError("CSST official batch residual ratio returned non-finite values.")
        self._last_truth_backend_used = "cpu_batch"
        return np.maximum(ratio, float(max(self.power_eps, 1.0e-30))).astype(np.float64)

    def _resolve_torch_truth_device(self) -> Any:
        import torch

        requested = str(self.truth_device).strip().lower()
        if requested in {"", "auto"}:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CSST truth_device='cuda' was requested but CUDA is not available.")
            return torch.device("cuda:0")
        if requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CSST truth_device={self.truth_device!r} was requested but CUDA is not available.")
            return torch.device(requested)
        return torch.device(requested)

    def _resolve_torch_truth_dtype(self) -> Any:
        import torch

        name = str(self.truth_dtype).strip().lower()
        if name in {"float64", "double"}:
            return torch.float64
        if name in {"float32", "single", ""}:
            return torch.float32
        raise ValueError(f"Unsupported CSST truth_dtype: {self.truth_dtype!r}")

    def _official_residual_ratio_batch_torch(self, emulator: Any, thetas: np.ndarray, k: np.ndarray) -> np.ndarray:
        import torch

        nltype = str(self.nonlinear_model).strip().lower()
        if not bool(self.p_cb) and not np.allclose(thetas[:, 7], 0.0, atol=1.0e-12):
            raise ValueError(
                "CSST residual batch output currently supports Pcb=True or zero-neutrino total matter. "
                "z2 fixes mnu=0 for the active 5D slice."
            )
        if nltype == "linear":
            model = emulator.Bkcb
        elif nltype == "halofit":
            model = emulator.Bkcb_halofit
        elif nltype == "hmcode2020":
            model = emulator.Bkcb_hmcode2020
        else:
            raise ValueError(f"Unsupported CSST nonlinear_model for residual output: {self.nonlinear_model!r}")

        device = self._resolve_torch_truth_device()
        if device.type != "cuda":
            raise RuntimeError("CSST cuda_torch truth backend requires a CUDA device.")
        dtype = self._resolve_torch_truth_dtype()
        ncosmo = self._normalized_theta_batch(thetas)
        with _CUDA_TRUTH_LOCK:
            if self._torch_residual_engine is None:
                self._torch_residual_engine = _TorchOfficialResidualRatioEngine(
                    model,
                    device=device,
                    dtype=dtype,
                    power_eps=float(self.power_eps),
                )
            ratio = self._torch_residual_engine.predict_ratio(ncosmo, k)
            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except Exception:
                pass
        if np.any(~np.isfinite(ratio)):
            raise RuntimeError("CSST official CUDA residual ratio returned non-finite values.")
        self._last_truth_backend_used = f"cuda_torch:{device}:{str(dtype).replace('torch.', '')}"
        return np.maximum(ratio, float(max(self.power_eps, 1.0e-30))).astype(np.float64)

    def _official_residual_ratio_batch_dispatch(
        self,
        emulator: Any,
        thetas: np.ndarray,
        k: np.ndarray,
        *,
        truth_backend: str | None = None,
    ) -> np.ndarray:
        backend = str(truth_backend or self.truth_backend or "cpu_batch").strip().lower()
        if backend in {"cuda", "cuda_torch", "torch_cuda"}:
            try:
                return self._official_residual_ratio_batch_torch(emulator, thetas, k)
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise
        if backend in {"auto", "cuda_if_available"}:
            try:
                import torch

                if torch.cuda.is_available():
                    return self._official_residual_ratio_batch_torch(emulator, thetas, k)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return self._official_residual_ratio_batch(emulator, thetas, k)
        if backend in {"cpu", "cpu_batch", "numpy"}:
            return self._official_residual_ratio_batch(emulator, thetas, k)
        raise ValueError(f"Unsupported CSST truth_backend: {truth_backend or self.truth_backend!r}")

    def run_hifi_residual(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "csst_official_residual",
    ) -> dict[str, Any]:
        """Return the official CSST residual ratio, not an original power spectrum."""

        del accuracy_config
        raw_theta = self._set_theta(theta)
        k = self._coerce_k_bins(k_bins)
        z = np.asarray([float(self.redshift)], dtype=np.float64)
        emulator = self._load_emulator()
        residual_ratio = self._official_residual_ratio(emulator, z, k)[0]
        return {
            "theta": raw_theta,
            "k_bins": k,
            "P_residual_hifi": residual_ratio.astype(np.float64),
            "log_residual_hifi": np.log(np.maximum(residual_ratio, float(max(self.power_eps, 1.0e-30)))).astype(
                np.float64
            ),
            "metadata": {
                "provider": "csst_official",
                "asset_version": str(asset_version),
                "redshift": float(self.redshift),
                "parameter_space": "csst8",
                "linear_model": str(self.linear_model),
                "nonlinear_model": str(self.nonlinear_model),
                "neutrino_mass_split": str(self.neutrino_mass_split),
                "p_cb": bool(self.p_cb),
                "target_kind": "csst_official_log_residual",
                "residual_semantics": "log(P_csst_nonlin / P_csst_official_anchor)",
                "anchor_power_provider": f"csst_official_{str(self.nonlinear_model).strip().lower()}",
                "anchor_power_label": str(self.nonlinear_model),
            },
        }

    def run_hifi_residuals(
        self,
        thetas_batch: Sequence[Sequence[float]] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "csst_official_residual_batch",
        truth_backend: str | None = None,
    ) -> dict[str, Any]:
        """Return official CSST residual ratios for a batch of cosmologies."""

        del accuracy_config
        raw_thetas = _coerce_csst_theta_batch(thetas_batch)
        self._check_theta_batch_bounds(raw_thetas)
        k = self._coerce_k_bins(k_bins)
        emulator = self._load_emulator()
        residual_ratio = self._official_residual_ratio_batch_dispatch(
            emulator,
            raw_thetas,
            k,
            truth_backend=truth_backend,
        )
        log_residual = np.log(np.maximum(residual_ratio, float(max(self.power_eps, 1.0e-30))))
        return {
            "theta": raw_thetas,
            "k_bins": k,
            "P_residual_hifi": residual_ratio.astype(np.float64),
            "log_residual_hifi": log_residual.astype(np.float64),
            "metadata": {
                "provider": "csst_official",
                "asset_version": str(asset_version),
                "redshift": float(self.redshift),
                "parameter_space": "csst8",
                "linear_model": str(self.linear_model),
                "nonlinear_model": str(self.nonlinear_model),
                "neutrino_mass_split": str(self.neutrino_mass_split),
                "p_cb": bool(self.p_cb),
                "target_kind": "csst_official_log_residual",
                "residual_semantics": "log(P_csst_nonlin / P_csst_official_anchor)",
                "anchor_power_provider": f"csst_official_{str(self.nonlinear_model).strip().lower()}",
                "anchor_power_label": str(self.nonlinear_model),
                "batch_vectorized": True,
                "truth_backend_requested": str(truth_backend or self.truth_backend),
                "truth_backend_used": str(self._last_truth_backend_used),
            },
        }

    def run_hifi_anchor(
        self,
        theta: Sequence[float] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "csst_official",
    ) -> dict[str, Any]:
        del accuracy_config
        raw_theta = self._set_theta(theta)
        k = self._coerce_k_bins(k_bins)
        z = np.asarray([float(self.redshift)], dtype=np.float64)
        emulator = self._load_emulator()
        p_nonlin = emulator.get_pknl(
            z=z,
            k=k,
            Pcb=bool(self.p_cb),
            lintype=str(self.linear_model),
            nltype=str(self.nonlinear_model),
            neutrino_mass_split=str(self.neutrino_mass_split),
        )[0]
        anchor = self._anchor_pk(raw_theta, k)
        target = np.asarray(p_nonlin, dtype=np.float64).reshape(-1)
        if target.shape != k.shape:
            raise RuntimeError(
                "CSST official emulator returned shape "
                f"{target.shape}, expected {k.shape}."
            )
        if np.any(~np.isfinite(target)):
            raise RuntimeError("CSST official emulator returned non-finite values.")
        return {
            "theta": raw_theta,
            "k_bins": k,
            "P_linear": anchor.astype(np.float64),
            "P_nonlin_hifi": np.maximum(target, float(max(self.power_eps, 1.0e-30))).astype(
                np.float64
            ),
            "metadata": {
                "provider": "csst_official",
                "asset_version": str(asset_version),
                "redshift": float(self.redshift),
                "parameter_space": "csst8",
                "linear_model": str(self.linear_model),
                "nonlinear_model": str(self.nonlinear_model),
                "neutrino_mass_split": str(self.neutrino_mass_split),
                "p_cb": bool(self.p_cb),
                "anchor_power_provider": str(self.anchor_provider_name),
                "anchor_power_label": str(getattr(self.anchor_provider, "power_label", "hmcode2020")),
                "anchor_semantics": "P_linear stores the configured logdiff anchor spectra.",
            },
        }

    def run_hifi_anchors(
        self,
        thetas_batch: Sequence[Sequence[float]] | np.ndarray,
        k_bins: Sequence[float] | np.ndarray,
        accuracy_config: Any | None = None,
        asset_version: str = "csst_official",
    ) -> list[dict[str, Any]]:
        thetas = np.asarray(thetas_batch, dtype=np.float64)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        return [
            self.run_hifi_anchor(
                theta=theta,
                k_bins=k_bins,
                accuracy_config=accuracy_config,
                asset_version=asset_version,
            )
            for theta in thetas
        ]


__all__ = [
    "CSSTCAMBNonlinearAnchorProvider",
    "CSSTDataProvider",
    "CSST_THETA_NAMES",
    "DEFAULT_CSST_THETA_BOUNDS",
    "DEFAULT_VENDOR_PATH",
    "csst_theta_bounds_as_array",
]
