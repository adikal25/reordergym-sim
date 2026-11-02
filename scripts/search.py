"""Evolutionary program search, scored on calm or on varied training scenes.

Both variants start from the same zero-shot programs and are judged on the same
held-out shocks, which neither ever sees.

Run: uv run --env-file .env python -u scripts/search.py --fitness calm --seed 0
     uv run --env-file .env python -u scripts/search.py --fitness varied --seed 0
"""

import argparse
import json
import random
import threading
from pathlib import Path
from statistics import mean

import anthropic

from reordergym.experiment import (
    WAREHOUSE,
    evaluate_on_test,
    llm_factory,
    print_paired_table,
    print_worst_case_gap,
    safe_costs,
    scenes,
    tune_baselines,
)
from reordergym.llm import (
    GenerationError,
    ProgramError,
    build_improve_prompt,
    build_prompt,
    write_policy,
)
from reordergym.scene import Family
from reordergym.search import Candidate, evolve
from reordergym.tuning import Params, PolicyFactory

TRAIN_FAMILY: dict[str, Family] = {"calm": "stable", "varied": "varied"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitness", choices=["calm", "varied"], required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--children", type=int, default=8)
    parser.add_argument("--seeds-dir", type=Path, default=Path("runs/zero_shot_20"))
    parser.add_argument("--seed", type=int, default=0, help="parent-selection random seed")
    parser.add_argument("--budget", type=float, default=15.0, help="max API spend, USD")
    args = parser.parse_args()
    out_dir = Path(f"runs/search_{args.fitness}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    family = TRAIN_FAMILY[args.fitness]
    fitness_scenes = scenes(family, 0, 10)
    print(f"Tuning baselines on 20 '{family}' training scenes...")
    policies: dict[str, tuple[PolicyFactory, Params]] = tune_baselines(scenes(family, 0, 20))

    def evaluate(code: str) -> float | None:
        costs = safe_costs(llm_factory(code), {}, fitness_scenes)
        return None if costs is None else mean(costs)

    print(f"\nScoring seed programs from {args.seeds_dir}/ on 10 '{family}' scenes...")
    seeds = []
    for path in sorted(args.seeds_dir.glob("program_*.py")):
        fitness = evaluate(path.read_text())
        if fitness is not None:
            seeds.append(Candidate(path.read_text(), fitness, f"seed {path.stem}"))
    print(f"  {len(seeds)} valid seeds, best {min(c.fitness for c in seeds):,.0f}")

    base_prompt = build_prompt(WAREHOUSE, demand_history_days=56, lead_time_history_len=10)
    client = anthropic.Anthropic()
    spend = {"usd": 0.0}
    lock = threading.Lock()

    def propose(a: Candidate, b: Candidate) -> str | None:
        with lock:
            if spend["usd"] >= args.budget:
                return None
        prompt = build_improve_prompt(base_prompt, [(a.code, a.fitness), (b.code, b.fitness)])
        try:
            program = write_policy(client, prompt)
        except GenerationError as error:
            print(f"    proposal failed: {error}")
            with lock:
                spend["usd"] += error.cost_usd
            return None
        except (ProgramError, anthropic.APIError) as error:
            print(f"    proposal failed: {error}")
            return None
        with lock:
            spend["usd"] += program.cost_usd
        return program.code

    best_per_round: list[Candidate] = [min(seeds, key=lambda c: c.fitness)]

    def on_round(round_number: int, population: list[Candidate]) -> None:
        best = min(population, key=lambda c: c.fitness)
        best_per_round.append(best)
        new = [c for c in population if c.origin == f"round {round_number}"]
        for i, child in enumerate(new):
            (out_dir / f"round{round_number}_child{i}.py").write_text(child.code)
        print(
            f"  round {round_number}: {len(new)} valid children, best so far "
            f"{best.fitness:,.0f} ({best.origin}), spent ${spend['usd']:.2f}"
        )

    print(f"\nEvolving: {args.rounds} rounds x {args.children} children...")
    population = evolve(
        seeds,
        propose,
        evaluate,
        rounds=args.rounds,
        children_per_round=args.children,
        rng=random.Random(args.seed),
        on_round=on_round,
    )

    # The best program after each round, judged on held-out shocks it never saw.
    seen: set[str] = set()
    for round_number, best in enumerate(best_per_round):
        if best.code not in seen:
            seen.add(best.code)
            policies[f"best after round {round_number}"] = (llm_factory(best.code), {})
    results = evaluate_on_test(policies)
    print_paired_table(results)
    print_worst_case_gap(results)

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "fitness": args.fitness,
                "api_cost_usd": spend["usd"],
                "population": [{"origin": c.origin, "fitness": c.fitness} for c in population],
                "best_per_round": [
                    {"origin": c.origin, "fitness": c.fitness} for c in best_per_round
                ],
                "test": results,
            },
            indent=2,
        )
    )
    print(f"\nAPI cost ${spend['usd']:.2f}. Saved to {out_dir}/")


if __name__ == "__main__":
    main()
