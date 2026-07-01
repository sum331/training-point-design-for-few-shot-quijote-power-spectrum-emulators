# z2quijote

`z2quijote` is the self-contained workspace for the Quijote residual-anchor
line. It keeps the stable runtime structure proven by the earlier Quijote
pipeline, but the active code now lives under the `z2quijote` package instead
of importing from historical version snapshots.

## Current Direction

The next active target is a CDM nonlinear-power residual, not direct Quijote
power:

\[
r_{\mathrm{CDM}}(\theta,k)
=
\log P_{\mathrm{truth}}^{\mathrm{CDM}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

The anchor is a CAMB-like CDM nonlinear-power source rather than Quijote itself.
Both the baseline and the proposed method should use the same anchor/logdiff
target so that the comparison is stable and has fewer calibrated moving parts.

The PPR component remains the first-stage design input because the accepted
PPR designs are tuned directly on the Quijote validation \(p68\) metric. Its
current role is to provide mature early-stage geometry for the residual-anchor
pipeline.

## Layout

- `src/z2quijote/`: active z2 package.
- `src/z2quijote/runtime_core/`: z2-owned M1/M2/M3 runtime core, copied from
  the stable Quijote framework and renamed into the z2 package namespace.
- `src/z2quijote/csst/`: z2-owned CSST fastmock provider used by the M3 bias
  term.
- `scripts/`: z2 entry scripts for PPR, tuning, validation, plotting, and
  operational runs.
- `configs/`: runtime-core configuration used by the z2 M3 adapter.
- `tests/`: z2 regression and smoke tests.
- `archive/`: non-active historical material kept only for audit or rollback.
- `DATA_LAYOUT.md`: G-drive data roots and hashes for large computational
  assets.

Runtime data and copied computational outputs belong under:

```text
data
```

## Runtime Commands

The active commands resolve z2 code from `src/z2quijote` and large data from
`data`:

```powershell
python versions/z2quijote/run.py show-config --config versions/z2quijote/config.yaml
python versions/z2quijote/run.py smoke --config versions/z2quijote/config.yaml
python versions/z2quijote/run.py run-comparison --config versions/z2quijote/config.yaml
python versions/z2quijote/run.py translation-harness
```

Historical data paths may still contain names such as `v2_quijote` because they
refer to copied G-drive assets. They should be treated as data provenance, not
as active code dependencies.

## Harness/Loop Entry

The project now embeds the lightweight harness/loop core under
`src/z2quijote/hloop/`. The manuscript translation workflow is wired into the
formal runner:

```powershell
python versions/z2quijote/run.py translation-harness
```

By default this checks the preserved Chinese source under
`docs/paper_manuscript_20260701/chinese_source/` against the current English
Markdown manuscript and writes reports to:

```text
versions/z2quijote/docs/translation_harness_reports
```

For a different task or draft pair, pass explicit paths:

```powershell
python versions/z2quijote/run.py translation-harness `
  --source path/to/source_cn.md `
  --candidate path/to/candidate_en.md `
  --out-dir path/to/reports
```
