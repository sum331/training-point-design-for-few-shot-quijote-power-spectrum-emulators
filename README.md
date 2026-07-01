# Training-point design for few-shot Quijote power-spectrum emulators using a standard-geometry bias field and active learning

This repository contains the public reproducibility package for the manuscript
**"Training-point design for few-shot Quijote power-spectrum emulators using a standard-geometry bias field and active learning"**.

## What is included

- `code/`: z2 Quijote source code, selected scripts, tests, and public configs.
- `data/validation/`: fixed LHS256 validation coordinates and validation manifest.
- `data/main_run/`: compact design files and summary outputs for the main
  Sobol64 versus proposed-method comparison.
- `data/standard_geometry_bias/`: processed standard-geometry bias-field
  products used by PPR.
- `data/ablation/`: machine-readable ablation summaries, design files, and
  condition-level summaries.
- `data/figures/`: figure source data, tables, figure manifests, and source
  bundles for the schematic figures.
- `data/figures/source_data/Fig12_kun_proxy_quijote_enrichment_*`: direct
  diagnostic source data for the KUN prior versus Quijote cold-start difficulty.
- `figures/`: final manuscript figures in PNG/PDF/SVG formats.
- `docs/`: manuscript draft, formula sources, and reference inventory.
- `environment/`: dependency list, environment report, and anchor protocol.

## Main result

The fixed-budget comparison uses 64 Quijote training points and a fixed
256-point LHS validation set. The primary metric is overall \(p68\) relative
error. In the included ablation table, the Sobol64 baseline has
\(p68=0.019040150781028685\), while the full method has
\(p68=0.014901304004758833\).

## Reproducing the reported tables

The reviewer-facing machine-readable ablation summary is:

```text
data/ablation/ablation_results.csv
data/ablation/ablation_results_manifest.json
```

The validation protocol is summarized in:

```text
data/validation/validation_manifest.json
```

## Data availability boundary

This package does not redistribute raw Quijote \(N\)-body products or the full
third-party KUN/CSST emulator data bundle. It includes compact design points,
processed standard-geometry bias-field products, validation coordinates, figure
source data, and summary metrics required to audit the manuscript claims.

## Repository slug

Suggested GitHub repository name:

```text
training-point-design-for-few-shot-quijote-power-spectrum-emulators
```
