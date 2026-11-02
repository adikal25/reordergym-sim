"""Tune the classical baselines on calm scenes, then test them on shocks. No API key needed.

Run: uv run python scripts/baselines.py
"""

from reordergym.experiment import (
    evaluate_on_test,
    print_paired_table,
    print_worst_case_gap,
    scenes,
    tune_baselines,
)


def main() -> None:
    print("Tuning baselines on 20 stable training scenes...")
    policies = tune_baselines(scenes("stable", 0, 20))
    results = evaluate_on_test(policies)
    print_paired_table(results)
    print_worst_case_gap(results)


if __name__ == "__main__":
    main()
