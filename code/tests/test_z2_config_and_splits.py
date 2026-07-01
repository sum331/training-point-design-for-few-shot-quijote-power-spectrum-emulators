from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from z2quijote.config import load_config
from z2quijote.acquisition import (
    _boundary_aware_preselect,
    _build_alc_imse_state,
    _pca_component_weight_details,
    _pca_component_weights,
    _weighted_alc_imse_score,
)
from z2quijote.csst_fastmock import _build_bias_k_weights
from z2quijote.direct_cdm import SyntheticDirectCDMOracle
from z2quijote.emulator import PCAGPDirectCDMEmulator
from z2quijote.parameter_space import boundary_distance
from z2quijote.resources import load_r2_seed_geometry
from z2quijote.sampling import theta_rows_key
from z2quijote.splits import build_split_bundle
from z2quijote.theta_transform import active_to_csst8_theta, active_to_quijote_theta


def test_config_forbids_hmcode_anchor_and_lofi(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    assert config.target.kind == "direct_cdm_logpk"
    assert config.target.anchor_mode == "none"
    assert config.resources.use_lofi is False
    assert config.evaluation.primary_metric == "overall_relative_error.p68"
    assert config.evaluation.primary_curve == "kwise_p68_relative_error"
    assert config.evaluation.report_metric_policy == "p68_only"

    bad_anchor = _config_payload(tmp_path)
    bad_anchor["target"]["anchor_mode"] = "hmcode2020"
    path = tmp_path / "bad_anchor.yaml"
    path.write_text(yaml.safe_dump(bad_anchor), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor_mode"):
        load_config(path)

    logdiff = _config_payload(tmp_path)
    logdiff["target"] = {
        "kind": "cdm_logdiff",
        "anchor_mode": "camb_cdm_hmcode2020",
        "power_eps": 1.0e-12,
    }
    logdiff["active_learning"]["candidate_source"] = "m3"
    logdiff["fastmock_bias"] = {
        "enabled": True,
        "provider": "csst",
        "vendor_path": str(tmp_path / "csstemu_official_full"),
        "checkbound": False,
        "fixed_w": -1.0,
        "fixed_wa": 0.0,
        "fixed_mnu": 0.0,
        "reference_as": 2.1e-9,
        "bias_weight": 1.0,
        "bias_band_weights": [0.5, 1.5, 1.0],
        "normalization": "p95",
        "normalization_probe_count": 8,
        "cache_decimals": 10,
    }
    path = tmp_path / "logdiff_fastmock.yaml"
    path.write_text(yaml.safe_dump(logdiff), encoding="utf-8")
    logdiff_config = load_config(path)
    assert logdiff_config.target.kind == "cdm_logdiff"
    assert logdiff_config.target.anchor_mode == "camb_cdm_hmcode2020"
    assert logdiff_config.fastmock_bias.enabled is True
    assert logdiff_config.fastmock_bias.fixed_w == pytest.approx(-1.0)
    assert logdiff_config.fastmock_bias.fixed_wa == pytest.approx(0.0)
    assert logdiff_config.fastmock_bias.fixed_mnu == pytest.approx(0.0)
    assert logdiff_config.fastmock_bias.bias_band_weights == (0.5, 1.5, 1.0)
    assert logdiff_config.fastmock_bias.truth_backend == "cpu_batch"
    assert logdiff_config.fastmock_bias.truth_dtype == "float32"
    assert logdiff_config.fastmock_bias.truth_chunk_size == 6144

    bad_lofi = _config_payload(tmp_path)
    bad_lofi["resources"]["use_lofi"] = True
    path = tmp_path / "bad_lofi.yaml"
    path.write_text(yaml.safe_dump(bad_lofi), encoding="utf-8")
    with pytest.raises(ValueError, match="LoFi"):
        load_config(path)

    bad_metric = _config_payload(tmp_path)
    bad_metric["evaluation"]["primary_metric"] = "overall_relative_error.p95"
    path = tmp_path / "bad_metric.yaml"
    path.write_text(yaml.safe_dump(bad_metric), encoding="utf-8")
    with pytest.raises(ValueError, match="primary_metric"):
        load_config(path)


def test_build_splits_are_disjoint_and_r2_is_geometry_only(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    bundle = build_split_bundle(config, force=True)
    roles = [
        bundle.seed_theta_unit,
        bundle.probe_theta_unit,
        bundle.pool_theta_unit,
        bundle.audit_theta_unit,
        bundle.sobol64_theta_unit,
        bundle.sobol128_theta_unit,
        bundle.sobol_tail_theta_unit,
    ]
    seen: set[tuple[float, ...]] = set()
    for role in roles:
        keys = set(theta_rows_key(role, decimals=config.splits.duplicate_decimals))
        assert seen.isdisjoint(keys)
        seen.update(keys)

    assert bundle.metadata["target_kind"] == "direct_cdm_logpk"
    assert bundle.metadata["use_lofi"] is False
    assert bundle.metadata["r2_seed"]["resource_use"] == "geometry_only"
    assert bundle.metadata["r2_seed"]["source_target_ignored"] is True
    assert bundle.metadata["r2_seed"]["legacy_target_metadata"]["source_training_target_kind"] == "logdiff"


def test_build_splits_can_use_fixed_lhs_audit_npz(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    audit_unit = np.array(
        [
            [0.11, 0.21, 0.31, 0.41, 0.51],
            [0.22, 0.32, 0.42, 0.52, 0.62],
            [0.33, 0.43, 0.53, 0.63, 0.73],
            [0.44, 0.54, 0.64, 0.74, 0.84],
            [0.15, 0.25, 0.35, 0.45, 0.55],
            [0.26, 0.36, 0.46, 0.56, 0.66],
            [0.37, 0.47, 0.57, 0.67, 0.77],
            [0.48, 0.58, 0.68, 0.78, 0.88],
        ],
        dtype=np.float64,
    )
    audit_path = tmp_path / "fixed_lhs_audit.npz"
    np.savez_compressed(audit_path, theta_unit=audit_unit)
    payload["splits"]["audit_source"] = "npz"
    payload["splits"]["audit_path"] = str(audit_path)
    payload["splits"]["audit_theta_unit_key"] = "theta_unit"
    path = tmp_path / "fixed_lhs_audit.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)
    bundle = build_split_bundle(config, force=True)

    np.testing.assert_allclose(bundle.audit_theta_unit, audit_unit)
    assert bundle.metadata["audit_source"]["source"] == "npz"
    assert bundle.metadata["audit_source"]["size"] == payload["splits"]["audit_size"]
    assert bundle.metadata["audit_source"]["path"] == str(audit_path.resolve())


def test_k_grid_can_use_native_npz_bins(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    k_source = np.asarray([0.005, 0.015, 0.05, 0.09, 0.7, 1.2, 2.8, 3.5], dtype=np.float64)
    k_path = tmp_path / "native_k.npz"
    np.savez_compressed(k_path, k_bins=k_source)
    payload["k_grid"] = {
        "source": "npz",
        "path": str(k_path),
        "key": "k_bins",
        "k_min": 0.01,
        "k_max": 3.0,
        "expected_count": 6,
        "bands": [
            {"name": "low", "k_min": 0.01, "k_max": 0.07},
            {"name": "mid", "k_min": 0.07, "k_max": 0.5},
            {"name": "focus_high", "k_min": 0.5, "k_max": 1.0},
            {"name": "tail", "k_min": 1.0, "k_max": 3.0},
        ],
    }
    payload["active_learning"]["pca_band_weights"] = [0.3, 1.35, 1.3, 1.05]
    path = tmp_path / "native_k_config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)

    np.testing.assert_allclose(config.k_grid.k_bins, k_source[1:-1])
    assert [band.count for band in config.k_grid.bands] == [2, 1, 1, 2]
    assert config.k_grid.band_edges == pytest.approx((0.07, 0.5, 1.0))


def test_csst_fixed5_parameter_box_reuses_r2_unit_geometry(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["parameter_space"] = {
        "name": "csst_fixed5_official_box",
        "theta_names": ["Omegab", "Omegam", "H0", "ns", "A"],
        "theta_bounds": {
            "Omegab": [0.04, 0.06],
            "Omegam": [0.24, 0.40],
            "H0": [60.0, 80.0],
            "ns": [0.92, 1.00],
            "A": [1.7, 2.5],
        },
    }
    payload["resources"]["r2_seed"]["theta_key"] = "theta_unit"
    path = tmp_path / "csst_fixed5.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)
    seed = load_r2_seed_geometry(config)
    assert seed.metadata["theta_semantics"] == "unit_geometry_renormalized_to_current_parameter_box"
    assert np.all(seed.theta_raw >= config.parameter_space.theta_bounds[:, 0][None, :])
    assert np.all(seed.theta_raw <= config.parameter_space.theta_bounds[:, 1][None, :])

    theta8 = active_to_csst8_theta(
        config.parameter_space,
        np.asarray([[0.05, 0.30, 67.0, 0.96, 2.1]], dtype=np.float64),
        np.asarray([0.01, 0.1], dtype=np.float64),
        calibrator=object(),
        fixed_w=-1.0,
        fixed_wa=0.0,
        fixed_mnu=0.0,
    )
    assert np.allclose(theta8[0], [0.05, 0.30, 67.0, 0.96, 2.1, -1.0, 0.0, 0.0])


def test_quijote_ordered_csstA_box_maps_to_csst_bounds(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["parameter_space"] = {
        "name": "quijote_csstA_matched_box",
        "theta_names": ["Omega_m", "Omega_b", "h", "n_s", "A"],
        "theta_bounds": {
            "Omega_m": [0.24, 0.40],
            "Omega_b": [0.04, 0.06],
            "h": [0.60, 0.80],
            "n_s": [0.92, 1.00],
            "A": [1.7, 2.5],
        },
    }
    path = tmp_path / "quijote_ordered_csstA.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(path)
    theta = np.asarray([[0.30, 0.05, 0.67, 0.96, 2.1]], dtype=np.float64)

    theta8 = active_to_csst8_theta(
        config.parameter_space,
        theta,
        np.asarray([0.01, 0.1], dtype=np.float64),
        calibrator=object(),
        fixed_w=-1.0,
        fixed_wa=0.0,
        fixed_mnu=0.0,
    )
    assert np.allclose(theta8[0], [0.05, 0.30, 67.0, 0.96, 2.1, -1.0, 0.0, 0.0])

    class _FakeCalibrator:
        reference_as = 2.1e-9

        def sigma8_for_as(self, theta, k_bins, *, primordial_as):
            return 0.7

    quijote_native = active_to_quijote_theta(
        config.parameter_space,
        theta,
        np.asarray([0.01, 0.1], dtype=np.float64),
        _FakeCalibrator(),
    )
    assert np.allclose(quijote_native[0], [0.30, 0.05, 0.67, 0.96, 0.7])


def test_boundary_aware_preselect_keeps_body_candidates(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    boundary = np.column_stack(
        [
            np.full(20, 0.001),
            np.linspace(0.1, 0.9, 20),
            np.linspace(0.2, 0.8, 20),
            np.linspace(0.3, 0.7, 20),
            np.linspace(0.4, 0.6, 20),
        ]
    )
    body = np.column_stack(
        [
            np.full(20, 0.5),
            np.linspace(0.2, 0.8, 20),
            np.linspace(0.25, 0.75, 20),
            np.linspace(0.3, 0.7, 20),
            np.linspace(0.35, 0.65, 20),
        ]
    )
    theta_unit = np.vstack([boundary, body])
    theta_raw = config.parameter_space.denormalize(theta_unit)
    scores = np.concatenate([np.linspace(100.0, 81.0, 20), np.linspace(20.0, 1.0, 20)])
    selected = _boundary_aware_preselect(config=config, theta_raw=theta_raw, scores=scores, count=16)
    selected_boundary = boundary_distance(theta_unit[selected]) < config.active_learning.boundary_guard_threshold
    assert int(selected_boundary.sum()) <= int(config.active_learning.boundary_fraction_cap * 16)
    assert int((~selected_boundary).sum()) > 0


def test_active_learning_pca_band_weights_are_validated(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["active_learning"]["alc_imse_weight"] = 0.5
    payload["active_learning"]["pca_band_weights"] = [0.25, 1.25, 1.0]
    payload["active_learning"]["pca_weight_function"] = "band_integral"
    payload["active_learning"]["pca_weight_transition_dex"] = 0.08
    path = tmp_path / "weighted.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(path)
    assert config.active_learning.alc_imse_weight == pytest.approx(0.5)
    assert config.active_learning.pca_band_weights == (0.25, 1.25, 1.0)
    assert config.active_learning.pca_weight_function == "band_integral"
    assert config.active_learning.pca_weight_transition_dex == pytest.approx(0.08)

    bad_payload = _config_payload(tmp_path)
    bad_payload["active_learning"]["pca_band_weights"] = [1.0, 1.0]
    bad_path = tmp_path / "bad_band_weights.yaml"
    bad_path.write_text(yaml.safe_dump(bad_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pca_band_weights"):
        load_config(bad_path)

    bad_function = _config_payload(tmp_path)
    bad_function["active_learning"]["pca_weight_function"] = "unknown"
    bad_function_path = tmp_path / "bad_weight_function.yaml"
    bad_function_path.write_text(yaml.safe_dump(bad_function), encoding="utf-8")
    with pytest.raises(ValueError, match="pca_weight_function"):
        load_config(bad_function_path)


def test_fastmock_bias_band_weights_are_validated_and_shape_k_weights(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["target"] = {
        "kind": "cdm_logdiff",
        "anchor_mode": "camb_cdm_nonlinear",
        "power_eps": 1.0e-12,
    }
    payload["active_learning"]["candidate_source"] = "m3"
    payload["fastmock_bias"] = {
        "enabled": True,
        "provider": "csst",
        "vendor_path": str(tmp_path / "csstemu_official_full"),
        "checkbound": False,
        "fixed_w": -1.0,
        "fixed_wa": 0.0,
        "fixed_mnu": 0.0,
        "reference_as": 2.1e-9,
        "bias_weight": 1.0,
        "bias_band_weights": [0.25, 1.0, 4.0],
        "normalization": "p68",
        "normalization_probe_count": 8,
        "cache_decimals": 10,
    }
    path = tmp_path / "bias_band_weights.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(path)
    assert config.fastmock_bias.bias_band_weights == (0.25, 1.0, 4.0)

    point_weights, details = _build_bias_k_weights(
        config,
        np.asarray(config.k_grid.k_bins, dtype=np.float64),
        np.asarray(config.fastmock_bias.bias_band_weights, dtype=np.float64),
    )
    assert float(np.sum(point_weights)) == pytest.approx(1.0)
    assert details["bias_statistic"] == "weighted_mean_relative_error_over_k"
    assert details["bias_k_weight_curve_band_means"][2] > details["bias_k_weight_curve_band_means"][0]

    bad_payload = dict(payload)
    bad_payload["fastmock_bias"] = dict(payload["fastmock_bias"])
    bad_payload["fastmock_bias"]["bias_band_weights"] = [1.0, 1.0]
    bad_path = tmp_path / "bad_bias_band_weights.yaml"
    bad_path.write_text(yaml.safe_dump(bad_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bias_band_weights"):
        load_config(bad_path)


def test_pca_weight_function_controls_segment_components(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    k_bins = np.asarray(config.k_grid.k_bins, dtype=np.float64)
    components = np.zeros((3, k_bins.shape[0]), dtype=np.float64)
    masks = [
        (k_bins >= 0.01) & (k_bins < 0.07),
        (k_bins >= 0.07) & (k_bins < 0.5),
        (k_bins >= 0.5) & (k_bins <= 3.0),
    ]
    for index, mask in enumerate(masks):
        components[index, mask] = 1.0 / np.sqrt(max(int(np.count_nonzero(mask)), 1))
    emulator = SimpleNamespace(
        pca=SimpleNamespace(components_=components),
        k_bins=k_bins,
    )
    common_active = replace(
        config.active_learning,
        pca_weight_function="smooth_logk_curve",
        pca_component_weight_normalize=False,
        pca_component_weight_min=0.01,
        pca_component_weight_max=100.0,
    )
    low_config = replace(
        config,
        active_learning=replace(common_active, pca_band_weights=(5.0, 1.0, 1.0)),
    )
    low_weights, low_details = _pca_component_weight_details(low_config, emulator)
    assert low_weights[0] > low_weights[1]
    assert low_weights[0] > low_weights[2]
    assert low_details["k_weight_curve_band_means"][0] > low_details["k_weight_curve_band_means"][1]

    tail_config = replace(
        config,
        active_learning=replace(common_active, pca_band_weights=(1.0, 1.0, 5.0)),
    )
    tail_weights, tail_details = _pca_component_weight_details(tail_config, emulator)
    assert tail_weights[2] > tail_weights[0]
    assert tail_weights[2] > tail_weights[1]
    assert tail_details["k_weight_curve_band_means"][2] > tail_details["k_weight_curve_band_means"][1]


def test_weighted_alc_imse_scores_are_finite(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    config = replace(
        config,
        active_learning=replace(
            config.active_learning,
            alc_imse_weight=0.7,
            alc_probe_weight_floor=0.05,
            pca_band_weights=(0.25, 1.35, 1.0),
        ),
    )
    oracle = SyntheticDirectCDMOracle(config)
    train_unit = np.asarray(
        [
            [0.12, 0.18, 0.24, 0.30, 0.36],
            [0.88, 0.18, 0.24, 0.30, 0.36],
            [0.20, 0.82, 0.28, 0.34, 0.40],
            [0.24, 0.30, 0.78, 0.38, 0.44],
            [0.28, 0.34, 0.40, 0.74, 0.48],
            [0.32, 0.38, 0.44, 0.50, 0.82],
        ],
        dtype=np.float64,
    )
    probe_unit = np.asarray(
        [
            [0.15, 0.20, 0.25, 0.30, 0.35],
            [0.75, 0.25, 0.35, 0.45, 0.55],
            [0.35, 0.75, 0.45, 0.55, 0.65],
            [0.45, 0.55, 0.75, 0.65, 0.25],
            [0.55, 0.65, 0.35, 0.75, 0.45],
            [0.65, 0.35, 0.55, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    candidate_unit = np.asarray(
        [
            [0.16, 0.22, 0.28, 0.34, 0.40],
            [0.70, 0.30, 0.40, 0.50, 0.60],
            [0.40, 0.70, 0.50, 0.60, 0.30],
        ],
        dtype=np.float64,
    )
    train_theta = config.parameter_space.denormalize(train_unit)
    probe_theta = config.parameter_space.denormalize(probe_unit)
    candidate_theta = config.parameter_space.denormalize(candidate_unit)
    train_log = oracle.evaluate(train_theta, config.k_grid.k_bins).log_pk
    emulator = PCAGPDirectCDMEmulator(config.parameter_space, config.model).fit(
        train_theta,
        train_log,
        config.k_grid.k_bins,
    )
    probe_error = np.linspace(0.01, 0.08, probe_theta.shape[0], dtype=np.float64)
    state = _build_alc_imse_state(
        config=config,
        emulator=emulator,
        probe_theta_raw=probe_theta,
        probe_error=probe_error,
    )
    scores = _weighted_alc_imse_score(config=config, candidate_theta_raw=candidate_theta, state=state)
    component_weights = _pca_component_weights(config, emulator)
    assert state.enabled is True
    assert scores.shape == (candidate_theta.shape[0],)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert float(np.max(scores)) > 0.0
    assert component_weights.shape[0] == emulator.pca.n_components_
    assert np.all(np.isfinite(component_weights))
    assert state.weight_details["function"] == "smooth_logk_curve"
    assert len(state.weight_details["k_weight_curve_band_means"]) == len(config.k_grid.bands)


def _write_config(tmp_path: Path) -> Path:
    payload = _config_payload(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _config_payload(tmp_path: Path) -> dict:
    seed_path = tmp_path / "r2_seed.npz"
    manifest_path = tmp_path / "r2_seed.json"
    if not seed_path.exists():
        theta = np.asarray(
            [
                [0.204, 0.0356, 0.604, 0.904, 0.704],
                [0.396, 0.0356, 0.604, 0.904, 0.704],
                [0.204, 0.0644, 0.604, 0.904, 0.704],
                [0.204, 0.0356, 0.796, 0.904, 0.704],
                [0.204, 0.0356, 0.604, 1.096, 0.704],
                [0.204, 0.0356, 0.604, 0.904, 0.896],
            ],
            dtype=np.float64,
        )
        unit = np.linspace(0.05, 0.95, theta.size, dtype=np.float64).reshape(theta.shape)
        np.savez_compressed(seed_path, theta_raw=theta, theta_unit=unit)
        manifest_path.write_text(
            json.dumps(
                {
                    "source_training_target_kind": "logdiff",
                    "source_target": "log_hi_minus_log_hmcode2020_anchor",
                    "source_runtime_target": "quijote_hmcode2020_logdiff_reconstructed_power",
                }
            ),
            encoding="utf-8",
        )
    return {
        "project_root": str(tmp_path),
        "data_root": str(tmp_path / "data"),
        "random_seed": 123,
        "target": {"kind": "direct_cdm_logpk", "anchor_mode": "none", "power_eps": 1.0e-12},
        "parameter_space": {
            "name": "quijote_bsq5_tight10_k001_3",
            "theta_names": ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"],
            "theta_bounds": {
                "Omega_m": [0.14, 0.46],
                "Omega_b": [0.026, 0.074],
                "h": [0.54, 0.86],
                "n_s": [0.84, 1.16],
                "sigma_8": [0.64, 0.96],
            },
        },
        "k_grid": {
            "bands": [
                {"name": "low", "k_min": 0.01, "k_max": 0.07, "count": 6},
                {"name": "mid", "k_min": 0.07, "k_max": 0.5, "count": 8},
                {"name": "tail", "k_min": 0.5, "k_max": 3.0, "count": 8},
            ]
        },
        "resources": {
            "use_lofi": False,
            "v2_root": str(tmp_path),
            "truth_generator": {
                "kind": "v2_direct_logpk_truth_generator",
                "path": str(tmp_path / "unused_truth.pkl"),
                "device": "cpu",
                "chunk_size": 8,
            },
            "r2_seed": {
                "mode": "geometry_only",
                "path": str(seed_path),
                "theta_key": "theta_raw",
                "manifest_path": str(manifest_path),
            },
            "current_active": {"enabled": False, "path": None, "theta_key": "theta_raw"},
        },
        "splits": {
            "seed_label": "test_seed",
            "probe_size": 6,
            "pool_size": 16,
            "audit_size": 8,
            "sobol64_size": 10,
            "sobol128_size": 16,
            "sobol_tail_size": 4,
            "duplicate_decimals": 12,
            "output_dir": "manifests",
        },
        "model": {
            "pca_components": 4,
            "gp_alpha": 1.0e-8,
            "gp_n_restarts_optimizer": 0,
            "normalize_y": True,
            "length_scale_initial": 0.25,
            "length_scale_bounds": [0.01, 20.0],
            "constant_value": 1.0,
            "constant_value_bounds": [0.01, 100.0],
        },
        "active_learning": {
            "active_points": 4,
            "batch_size": 2,
            "preselect_factor": 4,
            "probe_error_percentile": 68.0,
            "uncertainty_weight": 0.35,
            "train_distance_weight": 0.15,
            "boundary_risk_weight": 0.10,
            "diversity_weight": 0.45,
            "reduction_probe_anchors": 4,
            "reduction_length_scale": 0.20,
            "boundary_guard_threshold": 0.055,
            "boundary_fraction_cap": 0.35,
        },
        "evaluation": {
            "primary_metric": "overall_relative_error.p68",
            "primary_curve": "kwise_p68_relative_error",
            "report_metric_policy": "p68_only",
            "band_edges": [0.07, 0.5],
            "band_labels": ["low", "mid", "tail"],
            "output_dir": "runs",
            "save_predictions": False,
        },
        "oracle_kind": "synthetic_direct_cdm",
    }
