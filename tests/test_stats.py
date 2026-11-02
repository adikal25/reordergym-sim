import pytest

from reordergym.stats import bootstrap_mean_ci, compare_paired


def test_constant_difference_collapses_ci() -> None:
    reference = [10.0, 20.0, 30.0, 40.0]
    policy = [value - 5.0 for value in reference]
    result = compare_paired(policy, reference)
    assert result.n == 4
    assert result.mean_diff == -5.0
    assert (result.ci_low, result.ci_high) == (-5.0, -5.0)
    assert result.win_rate == 1.0


def test_win_rate_counts_ties_as_not_wins() -> None:
    result = compare_paired([1.0, 2.0, 3.0, 9.0], [2.0, 2.0, 4.0, 5.0])
    assert result.win_rate == 0.5


def test_same_seed_gives_identical_ci() -> None:
    values = [1.0, -2.0, 3.5, 0.5, -1.5, 2.0]
    assert bootstrap_mean_ci(values, seed=3) == bootstrap_mean_ci(values, seed=3)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0], [1.0, 2.0, 3.0])


def test_fewer_than_two_values_raise() -> None:
    with pytest.raises(ValueError):
        compare_paired([1.0], [2.0])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0])


def test_clearly_better_policy_ci_excludes_zero() -> None:
    reference = [100.0 + 3.0 * i for i in range(30)]
    noise = [(-1.0) ** i * 0.5 * (i % 4) for i in range(30)]
    policy = [r - 10.0 + e for r, e in zip(reference, noise, strict=True)]
    result = compare_paired(policy, reference)
    assert result.ci_high < 0.0


def test_symmetric_noise_ci_contains_zero() -> None:
    reference = [100.0] * 8
    policy = [100.0 + d for d in (-3.0, 3.0, -2.0, 2.0, -1.0, 1.0, -4.0, 4.0)]
    result = compare_paired(policy, reference)
    assert result.ci_low < 0.0 < result.ci_high
