# Anchor-generation protocol

The emulator target used in the manuscript is a residual-anchor target,

\[
r_Q(\theta,k)=\log P^{Q,\mathrm{nl}}(\theta,k)
-\log P^{\mathrm{CDM,nl}}_{\mathrm{anchor}}(\theta,k).
\]

The anchor is generated with CAMB as a CDM nonlinear matter power spectrum on
the same \(k\)-grid used by the Quijote emulator. All reported comparisons use
the same target transform, \(k\)-grid, parameter box, and LHS256 validation
coordinates. The package includes compact validation coordinates, design
points, processed bias-field products, and metrics; raw simulation products are
not redistributed.
