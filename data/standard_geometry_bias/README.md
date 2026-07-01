# Standard-geometry bias field data

This directory contains the compact processed products used to construct the
standard-geometry bias prior in the manuscript.

- `standard_geometry_bias_field.npz`: interpolated bias field and support arrays.
- `density_field.npz`: density/potential-facing field used by PPR.
- `interpolator_support.npz`: support points used by the reliability-weighted
  interpolation.
- `standard_geometry_bias_summary.json`: run-level statistics.

The raw auxiliary truth cache is not included because the paper only requires
the processed accepted-only field for the reported PPR design and diagnostics.
