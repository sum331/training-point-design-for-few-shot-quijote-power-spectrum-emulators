# Manuscript Reference Usage Status

Date: 2026-07-01

Policy: keep all 57 entries in `references.bib`, but use only 38 in the current z2 manuscript. The remaining 19 entries are retained in the library and marked as not used for the current manuscript.

## Summary

| Status | Count | Meaning |
|---|---:|---|
| `use_current_manuscript` | 38 | Allowed for the current formal manuscript. |
| `do_not_use_current_manuscript` | 19 | Keep in the library, but do not cite in the current manuscript unless scope changes. |

## Use In Current Manuscript: 38

### Core Method, Data, And Physics

| Citation key | Use tier | Reason |
|---|---|---|
| `VillaescusaNavarro2020Quijote` | core | Quijote high-fidelity simulation data source. |
| `Chen2025CSSTEmulator` | core | CSST/KUN auxiliary generator for acquisition-stage bias proxy. |
| `CSSTEmulatorDocs` | support | Official documentation/provenance for the vendored CSST emulator implementation. |
| `Lewis2000CAMB` | core | CAMB anchor computation. |
| `Mead2021HMcode2020` | core | HMcode nonlinear anchor correction. |
| `Smith2003Halofit` | support | HALOFIT/nonlinear-model lineage. |
| `Takahashi2012Halofit` | support | Revised HALOFIT/nonlinear-model lineage. |
| `EisensteinHu1998TransferFunction` | core | Transfer-function and matter-power physical background. |
| `CooraySheth2002HaloModel` | core | Halo-model / nonlinear clustering physical background. |

### Emulator And Computer-Experiment Method

| Citation key | Use tier | Reason |
|---|---|---|
| `RasmussenWilliams2006GPML` | core | Gaussian-process foundation. |
| `Sacks1989ComputerExperiments` | core | Computer-experiment design and emulation foundation. |
| `SantnerWilliamsNotz2018ComputerExperiments` | support | Modern computer-experiment textbook support. |
| `JolliffeCadima2016PCA` | core | PCA compression citation. |
| `Higdon2008HighDimensionalOutput` | core | High-dimensional simulator-output emulation. |
| `ContiOHagan2010MultiOutputEmulation` | core | Functional / multi-output emulator context. |
| `Pedregosa2011ScikitLearn` | implementation | Current production PCA/GP implementation stack. |

### Design, Geometry, Interpolation, And Active Learning

| Citation key | Use tier | Reason |
|---|---|---|
| `Sobol1967QMC` | core | Sobol baseline and low-discrepancy design. |
| `Owen1998ScrambledNets` | core | Scrambled/QMC design background. |
| `McKay1979LHS` | core | LHS validation/reference design. |
| `Johnson1990MinimaxMaximinDesigns` | core | Minimax/maximin design background. |
| `MorrisMitchell1995ExploratoryDesigns` | core | Space-filling computational-experiment design. |
| `Aurenhammer1991Voronoi` | core | Voronoi/Delaunay geometric foundation. |
| `Barber1996Qhull` | core | Qhull/Delaunay implementation foundation. |
| `Shepard1968Interpolation` | core | Irregular-support interpolation foundation. |
| `Du1999CVT` | core | CVT/particle-relaxation analogy for PPR. |
| `MacKay1992ActiveDataSelection` | core | Information-based active data selection. |
| `Cohn1996ActiveLearning` | core | Active learning with statistical models / ALC logic. |
| `GramacyLee2009AdaptiveDesign` | core | Sequential GP design for computer experiments. |
| `Virtanen2020SciPy` | implementation | QMC, geometry, interpolation, and optimization implementation stack. |

### Related Work And Future Work Kept Compact

| Citation key | Use tier | Reason |
|---|---|---|
| `Heitmann2009CosmicCalibration` | related_work | Classic cosmological emulator background. |
| `Lawrence2010CoyoteUniverse` | related_work | Matter-power emulator background. |
| `Knabenhans2021EuclidEmulator2` | related_work | Modern nonlinear matter-power emulator context. |
| `Angulo2021BACCO` | related_work | Modern large-scale-structure emulator context. |
| `Moran2023MiraTitanIV` | related_work | High-precision matter-power emulation context. |
| `KennedyOHagan2000MultiFidelity` | future_work | One foundational future-work reference for true multi-fidelity fitting. |
| `Ho2022MultifidelityMatterPower` | future_work | Matter-power-specific multi-fidelity GP future-work reference. |
| `Ho2023MFBox` | future_work | Matter-power-specific multifidelity/multiscale future-work reference. |
| `Harris2020NumPy` | implementation | Basic numerical implementation stack, if software citations are included. |

## Do Not Use In Current Manuscript: 19

These entries remain in `references.bib` for future projects, appendix expansion, or methods that are not part of the current z2 claim.

| Citation key | Mark | Reason |
|---|---|---|
| `Springel2005Gadget2` | `do_not_use_current_manuscript` | Simulation-engine lineage is not needed when citing Quijote as the data source. |
| `Springel2021Gadget4` | `do_not_use_current_manuscript` | Same as above; avoid distracting from Quijote/KUN data provenance. |
| `Jolliffe2002PCA` | `do_not_use_current_manuscript` | Redundant with `JolliffeCadima2016PCA` for current PCA citation needs. |
| `LeGratiet2014RecursiveCokriging` | `do_not_use_current_manuscript` | Recursive co-kriging is not implemented in current z2. |
| `Perdikaris2017NonlinearMultiFidelity` | `do_not_use_current_manuscript` | Nonlinear multi-fidelity fusion is not implemented in current z2. |
| `KuleszaTaskar2012DPP` | `do_not_use_current_manuscript` | Current method does not implement determinantal point processes. |
| `LiuWang2016SVGD` | `do_not_use_current_manuscript` | PPR is not SVGD and does not use Stein gradients. |
| `Paszke2019PyTorch` | `do_not_use_current_manuscript` | PyTorch is not needed for the current formal citation spine unless GPU details are emphasized. |
| `Gardner2018GPyTorch` | `do_not_use_current_manuscript` | GPyTorch is only support/provenance code, not the production z2 emulator. |
| `Hunter2007Matplotlib` | `do_not_use_current_manuscript` | Figure software citation is optional and not needed in the main manuscript set. |
| `ForemanMackey2016Corner` | `do_not_use_current_manuscript` | Corner plotting utility only. |
| `Waskom2021Seaborn` | `do_not_use_current_manuscript` | Plotting utility only. |
| `Heitmann2014CoyoteExtended` | `do_not_use_current_manuscript` | Related-work overlap; current set already keeps classic Coyote and modern emulator examples. |
| `Knabenhans2019EuclidEmulator` | `do_not_use_current_manuscript` | Superseded for current purposes by `Knabenhans2021EuclidEmulator2`. |
| `Arico2021BACCOBaryonification` | `do_not_use_current_manuscript` | Baryonification is not central to the current CDM residual-anchor result. |
| `Lawrence2017MiraTitanII` | `do_not_use_current_manuscript` | Related-work overlap; `Moran2023MiraTitanIV` is retained as the sharper modern Mira-Titan reference. |
| `Yang2025GokuSimulationSuite` | `do_not_use_current_manuscript` | Future generalized emulator/NN ecosystem, not current z2. |
| `Yang2026MultifidelityNN` | `do_not_use_current_manuscript` | Future multi-fidelity NN direction, not current z2. |
| `Yang2026GokuNEmu` | `do_not_use_current_manuscript` | Future NN emulator direction, not current PCA-GP/PPR result. |
