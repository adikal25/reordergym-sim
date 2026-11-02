"""Zero-shot LLM policies vs tuned classical baselines.

Run: uv run --env-file .env python scripts/zero_shot.py
Reuse saved programs (no API calls): uv run python scripts/zero_shot.py --reuse
More programs, own folder: ... zero_shot.py --n-programs 20 --out runs/zero_shot_20
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import anthropic

from reordergym.config import SkuParams, WarehouseConfig
from reordergym.generate import generate_scene
from reordergym.llm import ProgramError, build_prompt, load_policy, time_limit, write_policy
from reordergym.policies import adaptive_base_stock, base_stock, capped_base_stock, s_S
from reordergym.scene import Family, Scene
from reordergym.stats import compare_paired
from reordergym.tuning import (
    Params,
    PolicyFactory,
    grid_search,
    make_grid,
    mean_score,
    per_scene_costs,
)

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
FAMILIES: list[Family] = ["stable", "demand_spike", "supplier_disruption"]


def scenes(family: Family, first_seed: int, count: int) -> list[Scene]:
    return [generate_scene(WAREHOUSE, HORIZON, family, first_seed + i) for i in range(count)]


TRAIN = scenes("stable", 0, 20)
VALIDATION = scenes("stable", 10_000, 10)
TEST = {family: scenes(family, 20_000, 20) for family in FAMILIES}


def llm_factory(code: str) -> PolicyFactory:
    return lambda warehouse: load_policy(code, warehouse)


def safe_score(
    factory: PolicyFactory, params: Params, scene_set: list[Scene]
) -> list[float] | None:
    """Per-scene costs, or None if the program breaks the rules or runs too long."""
    try:
        with time_limit(120):
            return per_scene_costs(factory, params, scene_set, WAREHOUSE, WARMUP)
    except ProgramError as error:
        print(f"    invalid: {error}")
        return None


def generate_programs(n_programs: int, out_dir: Path) -> list[tuple[int, str | None, float]]:
    """Ask the LLM for n_programs programs. Returns (index, code or None, cost)."""
    print(f"\nAsking {n_programs} programs from the LLM...")
    prompt = build_prompt(WAREHOUSE, demand_history_days=56, lead_time_history_len=10)
    (out_dir / "prompt.txt").write_text(prompt)
    client = anthropic.Anthropic()

    def generate(i: int) -> tuple[int, str | None, float]:
        try:
            program = write_policy(client, prompt)
        except ProgramError as error:
            print(f"  program {i}: generation failed: {error}")
            return i, None, 0.0
        (out_dir / f"program_{i}.py").write_text(program.code)
        return i, program.code, program.cost_usd

    with ThreadPoolExecutor(max_workers=min(n_programs, 8)) as pool:
        return list(pool.map(generate, range(n_programs)))


BASELINES = ["base-stock", "(s,S)", "capped", "adaptive"]


def print_paired_table(results: dict[str, dict[Family, list[float] | None]]) -> None:
    """Each policy minus the best baseline of that family, scene by scene."""
    print("\nPaired vs best baseline per family (negative = cheaper; 95% bootstrap CI):")
    for family in FAMILIES:
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


def print_worst_case_gap(results: dict[str, dict[Family, list[float] | None]]) -> None:
    """Robustness: each policy's worst % gap to the best baseline, over all families."""
    best = {
        f: min(mean(c) for b in BASELINES if (c := results[b][f]) is not None) for f in FAMILIES
    }
    gaps = {
        name: max((mean(c) - best[f]) / best[f] for f, c in row.items() if c is not None)
        for name, row in results.items()
        if all(c is not None for c in row.values())
    }
    print("\nWorst-case gap to the best baseline across families (lower = more robust):")
    for name, gap in sorted(gaps.items(), key=lambda item: item[1]):
        print(f"  {name:<24} {gap:>+7.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse", action="store_true", help="load saved programs instead of calling the API"
    )
    parser.add_argument("--n-programs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/zero_shot"))
    args = parser.parse_args()
    out_dir: Path = args.out

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Tuning baselines on 20 stable training scenes...")
    zs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    candidates: dict[str, tuple[PolicyFactory, list[Params]]] = {
        "base-stock": (base_stock, make_grid(z=zs)),
        "(s,S)": (s_S, make_grid(z=zs, order_days=[1, 2, 4, 7])),
        "capped": (capped_base_stock, make_grid(z=zs, cap_days=[0.5, 0.75, 1, 1.5, 2, 3])),
        "adaptive": (adaptive_base_stock, make_grid(z=zs, window=[7, 14, 28, 56])),
    }
    policies: dict[str, tuple[PolicyFactory, Params]] = {}
    for name, (factory, grid) in candidates.items():
        result = grid_search(factory, grid, TRAIN, WAREHOUSE, WARMUP)
        policies[name] = (factory, result.params)
        print(f"  {name:<11} {result.params}")

    generated: list[tuple[int, str | None, float]]
    if args.reuse:
        files = sorted(out_dir.glob("program_*.py"))
        if not files:
            raise SystemExit(f"--reuse given but no program_*.py files in {out_dir}/")
        print(f"\nReusing {len(files)} saved programs from {out_dir}/")
        generated = [(int(f.stem.split("_")[1]), f.read_text(), 0.0) for f in files]
    else:
        generated = generate_programs(args.n_programs, out_dir)
    total_cost = sum(cost for _, _, cost in generated)
    print(f"  API cost: ${total_cost:.2f}")

    print("\nScoring programs on 10 stable validation scenes...")
    validation: dict[int, float] = {}
    for i, code, _ in generated:
        if code is None:
            continue
        costs = safe_score(llm_factory(code), {}, VALIDATION)
        print(f"  program {i}: {'invalid' if costs is None else f'{mean(costs):,.0f}'}")
        if costs is not None:
            validation[i] = mean(costs)
    for name, (factory, params) in policies.items():
        print(f"  {name:<11} {mean_score(factory, params, VALIDATION, WAREHOUSE, WARMUP):,.0f}")

    if validation:
        print(f"  best on validation: LLM #{min(validation, key=lambda i: validation[i])}")
    for i, code, _ in generated:
        if code is not None and i in validation:
            policies[f"LLM #{i}"] = (llm_factory(code), {})

    print("\nHeld-out test, 20 scenes per family (mean cost, lower is better):")
    print(f"{'':<24}" + "".join(f"{f:>22}" for f in FAMILIES))
    results: dict[str, dict[Family, list[float] | None]] = {}
    for name, (factory, params) in policies.items():
        row = {f: safe_score(factory, params, TEST[f]) for f in FAMILIES}
        results[name] = row
        cells = "".join(
            f"{'invalid' if v is None else f'{mean(v):,.0f}':>22}" for v in row.values()
        )
        print(f"{name:<24}" + cells)

    print_paired_table(results)
    print_worst_case_gap(results)

    (out_dir / "results.json").write_text(
        json.dumps(
            {"validation": validation, "test": results, "api_cost_usd": total_cost}, indent=2
        )
    )
    print(f"\nSaved programs and results to {out_dir}/")


if __name__ == "__main__":
    main()
