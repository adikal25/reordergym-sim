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

from reordergym.experiment import (
    WAREHOUSE,
    WARMUP,
    evaluate_on_test,
    llm_factory,
    print_paired_table,
    print_worst_case_gap,
    safe_costs,
    scenes,
    tune_baselines,
)
from reordergym.llm import GenerationError, build_prompt, write_policy
from reordergym.tuning import Params, PolicyFactory, mean_score

TRAIN = scenes("stable", 0, 20)
VALIDATION = scenes("stable", 10_000, 10)


def generate_programs(n_programs: int, out_dir: Path) -> list[tuple[int, str | None, float]]:
    """Ask the LLM for n_programs programs. Returns (index, code or None, cost)."""
    print(f"\nAsking {n_programs} programs from the LLM...")
    prompt = build_prompt(WAREHOUSE, demand_history_days=56, lead_time_history_len=10)
    (out_dir / "prompt.txt").write_text(prompt)
    client = anthropic.Anthropic()

    def generate(i: int) -> tuple[int, str | None, float]:
        try:
            program = write_policy(client, prompt)
        except GenerationError as error:
            print(f"  program {i}: generation failed: {error}")
            return i, None, error.cost_usd
        (out_dir / f"program_{i}.py").write_text(program.code)
        return i, program.code, program.cost_usd

    with ThreadPoolExecutor(max_workers=min(n_programs, 8)) as pool:
        return list(pool.map(generate, range(n_programs)))


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
    policies: dict[str, tuple[PolicyFactory, Params]] = tune_baselines(TRAIN)

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
        costs = safe_costs(llm_factory(code), {}, VALIDATION)
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

    results = evaluate_on_test(policies)

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
