from __future__ import annotations

import json
from pathlib import Path

from z2quijote.experiment import run_fair_comparison
from z2quijote.runtime_core.run_artifacts import run_process_path, run_results_path
from z2quijote.splits import build_split_bundle

from test_z2_config_and_splits import _write_config
from z2quijote.config import load_config
from z2quijote.direct_cdm import SyntheticDirectCDMOracle
from z2quijote.acquisition import select_z2_active_points


def test_synthetic_smoke_experiment_stays_direct_cdm(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    bundle = build_split_bundle(config, force=True)
    result = run_fair_comparison(config, split_bundle=bundle)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["target_kind"] == "direct_cdm_logpk"
    assert summary["anchor_mode"] == "none"
    assert summary["uses_lofi"] is False
    assert summary["active_selection"]["uses_lofi"] is False
    seed_budget = int(bundle.seed_theta_raw.shape[0])
    active_budget = summary["active_selection"]["active_points_selected"]
    total_budget = seed_budget + active_budget
    seed_name = f"ppr{seed_budget}"
    active_name = f"{seed_name}_plus_z2_active{active_budget}"
    hybrid_names = [name for name in summary["design_results"] if "_plus_z2_hybrid" in name]
    primary_name = hybrid_names[0] if hybrid_names else active_name
    assert f"sobol{total_budget}" in summary["design_results"]
    assert seed_name in summary["design_results"]
    assert f"{seed_name}_plus_sobol{active_budget}" in summary["design_results"]
    assert active_name in summary["design_results"]
    assert f"{primary_name}_vs_sobol{total_budget}" in summary["comparison"]
    assert f"{primary_name}_vs_{seed_name}" in summary["comparison"]
    dynamic_weights = summary["active_selection"]["dynamic_weight_state"]
    assert dynamic_weights["restored_after_completion"] is True
    assert dynamic_weights["lambda_bias_weight_dynamic"] is False
    assert dynamic_weights["after_restore"]["variance_band_weights"] == dynamic_weights["variance_band_weights_default"]
    assert dynamic_weights["after_restore"]["bias_band_weights"] == dynamic_weights["bias_band_weights_default"]
    assert "report_manifest_path" in summary
    assert result.summary_path.exists()
    assert run_results_path(result.run_dir, "active_learning_validation", "test_set_results.json").exists()
    assert run_results_path(result.run_dir, "fixed_budget_comparison", "test_set_results.json").exists()
    assert run_results_path(result.run_dir, "plots", "active_learning_validation_error.png").exists()
    assert run_results_path(result.run_dir, "plots", "z2_cdm_logdiff_summary_table.csv").exists()
    manifest_path = run_results_path(result.run_dir, "report_manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deprecated_plot_names = {
        "z2_cdm_logdiff_conclusion.png",
        "z2_cdm_logdiff_overall_p68_comparison.png",
        "z2_cdm_logdiff_improvement_fraction.png",
        "z2_cdm_logdiff_band_p68_comparison.png",
        "z2_cdm_logdiff_design_corner.png",
    }
    assert all(Path(path).name not in deprecated_plot_names for path in manifest.get("plot_paths", []))
    assert run_process_path(result.run_dir, "training_point_summary.json").exists()
    assert run_process_path(result.run_dir, "iteration_history.json").exists()
    for design in summary["design_results"].values():
        assert design["target_kind"] == "direct_cdm_logpk"
        assert design["metrics"]["target_kind"] == "direct_cdm_logpk"


def test_m3_candidate_source_does_not_select_pool_indices(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    config = _m3_tiny_config(config)
    bundle = build_split_bundle(config, force=True)
    oracle = SyntheticDirectCDMOracle(config)
    result = select_z2_active_points(
        config=config,
        oracle=oracle,
        seed_theta_raw=bundle.seed_theta_raw,
        probe_theta_raw=bundle.probe_theta_raw,
        pool_theta_raw=bundle.pool_theta_raw,
        k_bins=config.k_grid.k_bins,
    )
    assert result.report["candidate_source"] == "m3"
    assert result.selected_theta_raw.shape == (config.active_learning.active_points, 5)
    assert set(result.selected_pool_indices.tolist()) == {-2}
    assert result.report["rounds"][0]["m3_metadata"]["objective"] == "mid_high_weighted_sum"
    adapter = result.report["rounds"][0]["m3_metadata"]["z2_m3_adapter"]
    assert adapter["pca_weight_function"] == config.active_learning.pca_weight_function
    assert adapter["runtime_objective_mode"] == "mid_high_weighted_sum"
    dynamic_weights = result.report["dynamic_weight_state"]
    assert dynamic_weights["restored_after_completion"] is True
    assert dynamic_weights["lambda_bias_weight_dynamic"] is False


def _m3_tiny_config(config):
    from dataclasses import replace

    return replace(
        config,
        active_learning=replace(
            config.active_learning,
            candidate_source="m3",
            active_points=2,
            sobol_tail_reserve=1,
            batch_size=1,
            reduction_probe_anchors=4,
        ),
    )
