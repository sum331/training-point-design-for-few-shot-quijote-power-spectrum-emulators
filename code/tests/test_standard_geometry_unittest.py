from __future__ import annotations

import numpy as np

from z2quijote.standard_geometry import (
    BiasAccumulator,
    ReliabilityWeightedLocalInterpolator,
    StandardGeometryConfig,
    density_from_bias,
    draw_design_unit,
    draw_reference_unit,
)
from z2quijote.standard_geometry.geometry import accepted_mask, compute_geometry_batch, thresholds_from_geometry


def test_standard_geometry_batch_and_thresholds_are_finite() -> None:
    reference = draw_reference_unit(count=64, dim=5, seed=11)
    designs = [draw_design_unit(design_size=24, dim=5, seed=100, index=i) for i in range(6)]
    batches = [compute_geometry_batch(reference, design) for design in designs]

    thresholds = thresholds_from_geometry(
        batches,
        StandardGeometryConfig(tau_lambda=0.30, boundary_min=0.0),
    )
    assert thresholds.tau_lambda == 0.30
    assert thresholds.h_min <= thresholds.h_max
    assert thresholds.kappa_max > 0.0

    accepted = [accepted_mask(batch, thresholds) for batch in batches]
    assert all(mask.shape == (reference.shape[0],) for mask in accepted)
    assert sum(int(np.count_nonzero(mask)) for mask in accepted) >= 0


def test_bias_accumulator_estimates_accepted_mean_and_uncertainty() -> None:
    accumulator = BiasAccumulator(reference_size=4)
    accumulator.add(np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([True, False, True, True]))
    accumulator.add(np.asarray([3.0, 8.0, 5.0, 6.0]), np.asarray([True, True, False, True]))

    estimate = accumulator.estimate(usable_min_count=1, high_confidence_min_count=2)
    assert np.allclose(estimate.bias_mean, [2.0, 8.0, 3.0, 5.0])
    assert np.array_equal(estimate.accepted_count, [2, 1, 1, 2])
    assert np.array_equal(estimate.high_confidence, [True, False, False, True])
    assert np.all(np.isfinite(estimate.bias_se))


def test_bias_accumulator_indexed_updates_match_full_mask_updates() -> None:
    full = BiasAccumulator(reference_size=5)
    indexed = BiasAccumulator(reference_size=5)

    full.add(np.asarray([1.0, 2.0, np.nan, 4.0, 5.0]), np.asarray([True, False, True, True, False]))
    indexed.add_indices(np.asarray([0, 2, 3]), np.asarray([1.0, np.nan, 4.0]))
    full.add(np.asarray([3.0, 7.0, 9.0, 11.0, 13.0]), np.asarray([False, True, False, True, True]))
    indexed.add_indices(np.asarray([1, 3, 4]), np.asarray([7.0, 11.0, 13.0]))

    full_estimate = full.estimate(usable_min_count=1, high_confidence_min_count=2)
    indexed_estimate = indexed.estimate(usable_min_count=1, high_confidence_min_count=2)
    assert np.allclose(full_estimate.bias_mean, indexed_estimate.bias_mean, equal_nan=True)
    assert np.array_equal(full_estimate.accepted_count, indexed_estimate.accepted_count)
    assert np.array_equal(full_estimate.high_confidence, indexed_estimate.high_confidence)


def test_reliability_weighted_interpolator_and_density() -> None:
    theta = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    bias = np.asarray([1.0, 2.0, 3.0, 4.0, 2.5], dtype=np.float64)
    counts = np.asarray([20, 20, 20, 20, 5], dtype=np.int64)
    interpolator = ReliabilityWeightedLocalInterpolator(
        theta,
        bias,
        counts,
        min_count=5,
        high_confidence_count=20,
        neighbors=4,
        fallback_neighbors=5,
    )

    pred, confidence = interpolator.predict(np.asarray([[0.5, 0.5], [0.25, 0.25]], dtype=np.float64))
    assert pred.shape == (2,)
    assert confidence.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))

    density = density_from_bias(bias, alpha=1.0, clip_quantile=0.95)
    assert density.shape == bias.shape
    assert np.isclose(float(np.sum(density)), 1.0)
    assert np.all(density > 0.0)
