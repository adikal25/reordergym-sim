"""Shared experiment setup: one warehouse, fixed scene sets, baselines and result tables."""

from statistics import mean

from reordergym.config import SkuParams, WarehouseConfig
from reordergym.generate import generate_scene
from reordergym.llm import ProgramError, load_policy, time_limit
from reordergym.policies import adaptive_base_stock, base_stock, capped_base_stock, s_S
from reordergym.scene import Family, Scene
from reordergym.stats import compare_paired
from reordergym.tuning import Params, PolicyFactory, grid_search, make_grid, per_scene_costs

HORIZON = 365
WARMUP = 60
SKUS = tuple(
    SkuParams(
        unit_cost=2,
        holding_cost=0.05,
        lost_sales_penalty=5,
        fixed_order_cost=10,
        scrap_cost=1,
        max_order=1000,
        nominal_lead_time=2 + i % 4,
    )
    for i in range(8)
)
WAREHOUSE = WarehouseConfig(skus=SKUS, capacity=600)
TEST_FAMILIES: list[Family] = ["stable", "demand_spike", "supplier_disruption"]
BASELINES = ["base-stock", "(s,S)", "capped", "adaptive"]
Results = dict[str, dict[Family, list[float] | None]]


def scenes(family: Family, first_seed: int, count: int) -> list[Scene]:
    return [generate_scene(WAREHOUSE, HORIZON, family, first_seed + i) for i in range(count)]


# Seed ranges: training 0+, validation 10_000+, held-out test 20_000+.
TEST = {family: scenes(family, 20_000, 20) for family in TEST_FAMILIES}


def llm_factory(code: str) -> PolicyFactory:
    return lambda warehouse: load_policy(code, warehouse)


def safe_costs(
    factory: PolicyFactory, params: Params, scene_set: list[Scene]
) -> list[float] | None:
    """Per-scene costs, or None if the program breaks the rules or runs too long."""
    try:
        with time_limit(120):
            return per_scene_costs(factory, params, scene_set, WAREHOUSE, WARMUP)
    except ProgramError as error:
        print(f"    invalid: {error}")
        return None


def tune_baselines(train: list[Scene]) -> dict[str, tuple[PolicyFactory, Params]]:
    zs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    candidates: dict[str, tuple[PolicyFactory, list[Params]]] = {
        "base-stock": (base_stock, make_grid(z=zs)),
        "(s,S)": (s_S, make_grid(z=zs, order_days=[1, 2, 4, 7])),
        "capped": (capped_base_stock, make_grid(z=zs, cap_days=[0.5, 0.75, 1, 1.5, 2, 3])),
        "adaptive": (adaptive_base_stock, make_grid(z=zs, window=[7, 14, 28, 56])),
    }
    tuned: dict[str, tuple[PolicyFactory, Params]] = {}
    for name, (factory, grid) in candidates.items():
        result = grid_search(factory, grid, train, WAREHOUSE, WARMUP)
        tuned[name] = (factory, result.params)
        print(f"  {name:<11} {result.params}")
    return tuned


def evaluate_on_test(policies: dict[str, tuple[PolicyFactory, Params]]) -> Results:
    print("\nHeld-out test, 20 scenes per family (mean cost, lower is better):")
    print(f"{'':<24}" + "".join(f"{f:>22}" for f in TEST_FAMILIES))
    results: Results = {}
    for name, (factory, params) in policies.items():
        row = {f: safe_costs(factory, params, TEST[f]) for f in TEST_FAMILIES}
        results[name] = row
        cells = "".join(
            f"{'invalid' if v is None else f'{mean(v):,.0f}':>22}" for v in row.values()
        )
        print(f"{name:<24}" + cells)
    return results


def print_paired_table(results: Results) -> None:
    """Each policy minus the best baseline of that family, scene by scene."""
    print("\nPaired vs best baseline per family (negative = cheaper; 95% bootstrap CI):")
    for family in TEST_FAMILIES:
        baseline_costs = {
            name: costs for name in BASELINES if (costs := results[name][family]) is not None
        }
        best = min(baseline_costs, key=lambda name: mean(baseline_costs[name]))
        print(f"  {family} (reference: {best})")
        for name, row in results.items():
            costs = row[family]
            if name == best or costs is None:
                continue
            c = compare_paired(costs, baseline_costs[best])
            verdict = "better" if c.ci_high < 0 else "worse" if c.ci_low > 0 else "unclear"
            print(
                f"    {name:<24} {c.mean_diff:>+9,.0f}  [{c.ci_low:>+9,.0f}, {c.ci_high:>+9,.0f}]"
                f"  wins {c.win_rate:>4.0%}  {verdict}"
            )


def print_worst_case_gap(results: Results) -> None:
    """Robustness: each policy's worst % gap to the best baseline, over all families."""
    best = {
        f: min(mean(c) for b in BASELINES if (c := results[b][f]) is not None)
        for f in TEST_FAMILIES
    }
    gaps = {
        name: max((mean(c) - best[f]) / best[f] for f, c in row.items() if c is not None)
        for name, row in results.items()
        if all(c is not None for c in row.values())
    }
    print("\nWorst-case gap to the best baseline across families (lower = more robust):")
    for name, gap in sorted(gaps.items(), key=lambda item: item[1]):
        print(f"  {name:<24} {gap:>+7.1%}")
