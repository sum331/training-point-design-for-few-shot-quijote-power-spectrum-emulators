# Residual Anchor Bridge Contract

This file records the implementation target for the next connection step
between `source_snapshots/r2_v2_ppr` and `source_snapshots/v2_quijote`.

## Target

Use the same residual/logdiff target for the baseline and the proposed method:

\[
r_{\mathrm{CDM}}(\theta,k)
=
\log P_{\mathrm{truth}}^{\mathrm{CDM}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

The anchor must be a CDM nonlinear-power source from the CAMB/HMCode-like path,
not a Quijote training label. The current copied configs use:

```yaml
representation:
  anchor_mode: hmcode2020
  transform_family: logdiff

extensions:
  quijote:
    anchor_provider: camb_cdm_hmcode2020_sigma8_calibrated
```

and the R2-v2/PPR side uses:

```yaml
sources:
  anchor:
    matter_power_var: delta_cdm
  training_target: logdiff
```

## Accepted PPR Input

The current accepted first-stage design is the R2-v2 rank-body logdiff result
selected by \(p68\):

```text
data/source_data/r2_v2_ppr/artifacts/lofi_relaxation/ordered_p68_tuning_sobol64_cdm_logdiff_rankbody_20260602/51_adaptive_coord_joint_r01_strength_i001_s5em05/strength_0p0180219/blend_lambda_1/lofi_design.npz
```

Its recorded metrics are:

- overall \(p68\): `0.007348334620203757`;
- baseline \(p68\): `0.007893924440069328`;
- improvement fraction: `0.0691151560934854`.

## Connection Points

R2-v2/PPR:

- `source_snapshots/r2_v2_ppr/src/r2_multi_al/quijote_variance.py`
- `source_snapshots/r2_v2_ppr/src/r2_multi_al/quijote_validation_scoring.py`
- `source_snapshots/r2_v2_ppr/scripts/export_best64_to_v2_quijote.py`
- `source_snapshots/r2_v2_ppr/src/r2_multi_al/paths.py`

v2 Quijote:

- `source_snapshots/v2_quijote/src/representation.py`
- `source_snapshots/v2_quijote/src/quijote_gp_data_provider.py`
- `source_snapshots/v2_quijote/src/evaluation/active_learning_validation.py`
- `source_snapshots/v2_quijote/src/evaluation/comparison_report.py`
- `source_snapshots/v2_quijote/scripts/plotting/`
- `source_snapshots/v2_quijote/scripts/tuning/run_quijote_raw_bank_validation.py`

## Non-goals

- Do not continue optimizing the archived direct-CDM z2 M3 selector as the main
  path.
- Do not compare a direct-logP method against a residual/logdiff baseline.
- Do not let one side use a different anchor or target transform from the other
  side.
