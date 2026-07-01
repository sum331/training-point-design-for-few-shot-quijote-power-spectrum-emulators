from __future__ import annotations

from typing import Any

import numpy as np

from .parameter_space import ParameterSpace


QUIJOTE_BSQ5_NAMES: tuple[str, ...] = ("Omega_m", "Omega_b", "h", "n_s", "sigma_8")
QUIJOTE_CSSTA_NAMES: tuple[str, ...] = ("Omega_m", "Omega_b", "h", "n_s", "A")
CSST_FIXED5_NAMES: tuple[str, ...] = ("Omegab", "Omegam", "H0", "ns", "A")


def parameter_space_family(parameter_space: ParameterSpace) -> str:
    names = tuple(str(name) for name in parameter_space.theta_names)
    if names == QUIJOTE_BSQ5_NAMES:
        return "quijote_bsq5"
    if names == QUIJOTE_CSSTA_NAMES:
        return "quijote_csstA"
    if names == CSST_FIXED5_NAMES:
        return "csst_fixed5"
    raise ValueError(
        "Unsupported z2 parameter-space theta_names. Expected "
        f"{QUIJOTE_BSQ5_NAMES!r}, {QUIJOTE_CSSTA_NAMES!r}, or {CSST_FIXED5_NAMES!r}, got {names!r}."
    )


def active_to_quijote_theta(
    parameter_space: ParameterSpace,
    theta_raw: np.ndarray,
    k_bins: np.ndarray,
    calibrator: Any,
) -> np.ndarray:
    theta = _coerce_theta(theta_raw, parameter_space.dim)
    family = parameter_space_family(parameter_space)
    if family == "quijote_bsq5":
        return theta.astype(np.float64)
    if family == "quijote_csstA":
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        rows: list[np.ndarray] = []
        for row in theta:
            omegam, omegab, h, ns, amp_1e9_as = (float(value) for value in row)
            if omegam <= omegab:
                raise ValueError("Quijote-CSST matched theta requires Omega_m > Omega_b.")
            if h <= 0.0 or amp_1e9_as <= 0.0:
                raise ValueError("Quijote-CSST matched theta requires h and A to be positive.")
            background_theta = np.asarray([omegam, omegab, h, ns, 1.0], dtype=np.float64)
            reference_sigma8 = float(
                calibrator.sigma8_for_as(
                    background_theta,
                    k,
                    primordial_as=float(calibrator.reference_as),
                )
            )
            target_as = amp_1e9_as * 1.0e-9
            sigma8 = reference_sigma8 * np.sqrt(target_as / float(calibrator.reference_as))
            rows.append(np.asarray([omegam, omegab, h, ns, sigma8], dtype=np.float64))
        return np.vstack(rows).astype(np.float64)

    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    rows: list[np.ndarray] = []
    for row in theta:
        omegab, omegam, h0, ns, amp_1e9_as = (float(value) for value in row)
        if omegam <= omegab:
            raise ValueError("CSST fixed-5 theta requires Omegam > Omegab.")
        if h0 <= 0.0 or amp_1e9_as <= 0.0:
            raise ValueError("CSST fixed-5 theta requires H0 and A to be positive.")
        h = h0 / 100.0
        background_theta = np.asarray([omegam, omegab, h, ns, 1.0], dtype=np.float64)
        reference_sigma8 = float(
            calibrator.sigma8_for_as(
                background_theta,
                k,
                primordial_as=float(calibrator.reference_as),
            )
        )
        target_as = amp_1e9_as * 1.0e-9
        sigma8 = reference_sigma8 * np.sqrt(target_as / float(calibrator.reference_as))
        rows.append(np.asarray([omegam, omegab, h, ns, sigma8], dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


def active_to_csst8_theta(
    parameter_space: ParameterSpace,
    theta_raw: np.ndarray,
    k_bins: np.ndarray,
    calibrator: Any,
    *,
    fixed_w: float,
    fixed_wa: float,
    fixed_mnu: float,
) -> np.ndarray:
    theta = _coerce_theta(theta_raw, parameter_space.dim)
    family = parameter_space_family(parameter_space)
    rows: list[np.ndarray] = []
    if family == "csst_fixed5":
        for row in theta:
            omegab, omegam, h0, ns, amp_1e9_as = (float(value) for value in row)
            rows.append(
                np.asarray(
                    [omegab, omegam, h0, ns, amp_1e9_as, fixed_w, fixed_wa, fixed_mnu],
                    dtype=np.float64,
                )
            )
        return np.vstack(rows).astype(np.float64)
    if family == "quijote_csstA":
        for row in theta:
            omegam, omegab, h, ns, amp_1e9_as = (float(value) for value in row)
            rows.append(
                np.asarray(
                    [omegab, omegam, 100.0 * h, ns, amp_1e9_as, fixed_w, fixed_wa, fixed_mnu],
                    dtype=np.float64,
                )
            )
        return np.vstack(rows).astype(np.float64)

    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    for row in theta:
        omegam, omegab, h, ns, _ = (float(value) for value in row)
        target_as = float(calibrator.target_as_for_sigma8(row, k))
        rows.append(
            np.asarray(
                [omegab, omegam, 100.0 * h, ns, target_as * 1.0e9, fixed_w, fixed_wa, fixed_mnu],
                dtype=np.float64,
            )
        )
    return np.vstack(rows).astype(np.float64)


def quijote_to_active_theta(
    parameter_space: ParameterSpace,
    quijote_theta_raw: np.ndarray,
    k_bins: np.ndarray,
    calibrator: Any,
) -> np.ndarray:
    theta = _coerce_theta(quijote_theta_raw, 5)
    family = parameter_space_family(parameter_space)
    if family == "quijote_bsq5":
        return theta.astype(np.float64)
    if family == "quijote_csstA":
        k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
        rows: list[np.ndarray] = []
        for row in theta:
            omegam, omegab, h, ns, _ = (float(value) for value in row)
            target_as = float(calibrator.target_as_for_sigma8(row, k))
            rows.append(np.asarray([omegam, omegab, h, ns, target_as * 1.0e9], dtype=np.float64))
        return np.vstack(rows).astype(np.float64)

    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    rows: list[np.ndarray] = []
    for row in theta:
        omegam, omegab, h, ns, _ = (float(value) for value in row)
        target_as = float(calibrator.target_as_for_sigma8(row, k))
        rows.append(np.asarray([omegab, omegam, 100.0 * h, ns, target_as * 1.0e9], dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


def _coerce_theta(theta_raw: np.ndarray, dim: int) -> np.ndarray:
    theta = np.asarray(theta_raw, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    if theta.ndim != 2 or theta.shape[1] != int(dim):
        raise ValueError(f"theta_raw must have shape [N,{int(dim)}], got {theta.shape}.")
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta_raw contains non-finite values.")
    return theta.astype(np.float64)
