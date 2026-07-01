# z2quijote Data Layout

Large computational data is stored outside the repository under:

```text
data/source_data
```

## Direct-CDM Fallback Assets

These assets support the archived O1 direct-CDM fallback only.

| Asset | Path | Bytes | SHA256 |
|---|---|---:|---|
| direct-logP truth generator | `data/source_data/v2_quijote/artifacts/quijote/gp_surrogates/quijote_bsq_z0_full_directlogpk_truth_generator.pkl` | 67625967 | `BE16A7FBEC22597129CC008DD125F46D021A66CF4AA51258719D79EA5E2F373B` |
| legacy 512-point GP surrogate | `data/source_data/v2_quijote/artifacts/quijote/gp_surrogates/quijote_bsq_z0_gp.pkl` | 25682820 | `35B2B929C0A639166EAF1D73B0747BA1EDD88693E9996E95AB55413231614268` |
| official-linear generator | `data/source_data/v2_quijote/artifacts/quijote/gp_surrogates/quijote_bsq_z0_official_linear_directlogpk_generator.pkl` | 67567821 | `E5051469ED3D32FA4CC4B657E2D949274DAB13F3305080F1549BB32476F326F2` |
| raw-bank cache | `data/source_data/raw_bank/artifacts/quijote/cache/quijote_bsq_z0_bank.npz` | 104816692 | `570E537309C7912E3B2C61238FEC09CC221E35DD6CA0DEC35163AD33068A6462` |

## R2-v2 PPR Accepted Result

The accepted PPR64 result is the best row from the \(p68\)-tuned rank-body
logdiff run:

```text
data/source_data/r2_v2_ppr/artifacts/lofi_relaxation/ordered_p68_tuning_sobol64_cdm_logdiff_rankbody_20260602
```

Key copied files:

| Asset | Path | Bytes | SHA256 |
|---|---|---:|---|
| best parameters | `data/source_data/r2_v2_ppr/artifacts/lofi_relaxation/ordered_p68_tuning_sobol64_cdm_logdiff_rankbody_20260602/best_new_scheme_parameters.json` | 4528 | `CA837FEE6386E0DD58337C34B75C715C15B91739CAB36F4899C6122630A17CC8` |
| best design | `data/source_data/r2_v2_ppr/artifacts/lofi_relaxation/ordered_p68_tuning_sobol64_cdm_logdiff_rankbody_20260602/51_adaptive_coord_joint_r01_strength_i001_s5em05/strength_0p0180219/blend_lambda_1/lofi_design.npz` | 7841 | `52F5458B6A9C637D03877487EF538C12CB1CB32617F4A4EAC6B36F5AF32A8FEC` |

The associated validation summary is also copied:

```text
data/source_data/r2_v2_ppr/artifacts/lofi_relaxation/ordered_p68_tuning_sobol64_cdm_logdiff_rankbody_20260602/51_adaptive_coord_joint_r01_strength_i001_s5em05/strength_0p0180219/blend_lambda_1/quijote_validation/quijote_validation_comparison_summary.json
```

The original run reported:

- best stage: `51_adaptive_coord_joint_r01_strength_i001_s5em05`;
- best label: `strength_0p0180219`;
- blend lambda: `1`;
- overall \(p68\): `0.007348334620203757`;
- baseline \(p68\): `0.007893924440069328`;
- improvement fraction: `0.0691151560934854`.
