"""Zero-shot LLM policies vs tuned classical baselines.

Run: uv run --env-file .env python scripts/zero_shot.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from reordergym.config import SkuParams, WarehouseConfig
from reordergym.generate import generate_scene
from reordergym.llm import ProgramError, build_prompt, load_policy, time_limit, write_policy
from reordergym.policies import adaptive_base_stock, base_stock, capped_base_stock, s_S
from reordergym.scene import Family, Scene
from reordergym.tuning import Params, PolicyFactory, grid_search, make_grid, mean_score

N_PROGRAMS = 4
HORIZON = 365
WARMUP = 60
OUT_DIR = Path("runs/zero_shot")

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


def safe_score(factory: PolicyFactory, params: Params, scene_set: list[Scene]) -> float | None:
    """Mean cost, or None if the program breaks the rules or runs too long."""
    try:
        with time_limit(120):
            return mean_score(factory, params, scene_set, WAREHOUSE, WARMUP)
    except ProgramError as error:
        print(f"    invalid: {error}")
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"\nAsking {N_PROGRAMS} programs from the LLM...")
    prompt = build_prompt(WAREHOUSE, demand_history_days=56, lead_time_history_len=10)
    (OUT_DIR / "prompt.txt").write_text(prompt)
    client = anthropic.Anthropic()

    def generate(i: int) -> tuple[int, str | None, float]:
        try:
            program = write_policy(client, prompt)
        except ProgramError as error:
            print(f"  program {i}: generation failed: {error}")
            return i, None, 0.0
        (OUT_DIR / f"program_{i}.py").write_text(program.code)
        return i, program.code, program.cost_usd

    with ThreadPoolExecutor(max_workers=N_PROGRAMS) as pool:
        generated = list(pool.map(generate, range(N_PROGRAMS)))
    total_cost = sum(cost for _, _, cost in generated)
    print(f"  API cost: ${total_cost:.2f}")

    print("\nScoring programs on 10 stable validation scenes...")
    validation: dict[int, float] = {}
    for i, code, _ in generated:
        if code is None:
            continue
        score = safe_score(llm_factory(code), {}, VALIDATION)
        print(f"  program {i}: {'invalid' if score is None else f'{score:,.0f}'}")
        if score is not None:
            validation[i] = score
    for name, (factory, params) in policies.items():
        print(f"  {name:<11} {mean_score(factory, params, VALIDATION, WAREHOUSE, WARMUP):,.0f}")

    if validation:
        best = min(validation, key=lambda i: validation[i])
        best_code = next(code for i, code, _ in generated if i == best)
        assert best_code is not None
        policies[f"LLM best-of-{N_PROGRAMS} (#{best})"] = (llm_factory(best_code), {})
        first = min(validation)
        first_code = next(code for i, code, _ in generated if i == first)
        assert first_code is not None
        policies[f"LLM single (#{first})"] = (llm_factory(first_code), {})

    print("\nHeld-out test, 20 scenes per family (mean cost, lower is better):")
    print(f"{'':<24}" + "".join(f"{f:>22}" for f in FAMILIES))
    results: dict[str, dict[Family, float | None]] = {}
    for name, (factory, params) in policies.items():
        row = {f: safe_score(factory, params, TEST[f]) for f in FAMILIES}
        results[name] = row
        cells = "".join(f"{'invalid' if v is None else f'{v:,.0f}':>22}" for v in row.values())
        print(f"{name:<24}" + cells)

    (OUT_DIR / "results.json").write_text(
        json.dumps(
            {"validation": validation, "test": results, "api_cost_usd": total_cost}, indent=2
        )
    )
    print(f"\nSaved programs and results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
