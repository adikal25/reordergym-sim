"""Paired comparison statistics.

Comparisons are "paired" because every policy runs on the same scenes: for each scene we
subtract the reference cost from the policy cost, then summarise those per-scene differences.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_diff: float  # mean of (policy - reference); negative = policy is cheaper
    ci_low: float  # bootstrap CI of the mean difference
    ci_high: float
    win_rate: float  # fraction of scenes where policy cost < reference cost


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic given seed."""
    if len(values) < 2:
        raise ValueError("Need at least 2 values for a bootstrap CI")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(n_resamples, len(data)))
    resampled_means = data[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(resampled_means, [tail, 100.0 - tail])
    return float(low), float(high)


def compare_paired(
    policy_costs: Sequence[float],
    reference_costs: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Compare policy to reference scene by scene."""
    if len(policy_costs) != len(reference_costs):
        raise ValueError("policy_costs and reference_costs must have the same length")
    if len(policy_costs) < 2:
        raise ValueError("Need at least 2 scenes to compare")
    policy = np.asarray(policy_costs, dtype=float)
    reference = np.asarray(reference_costs, dtype=float)
    diffs = policy - reference
    ci_low, ci_high = bootstrap_mean_ci(
        diffs.tolist(), confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    return PairedComparison(
        n=len(diffs),
        mean_diff=float(diffs.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        win_rate=float((policy < reference).mean()),
    )
