"""Minimal evolutionary program search (FunSearch-style).

Each round: pick two parents from the best programs so far, ask for an improved child,
score it on fixed training scenes, and add it to the population if it is valid.
"""

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    code: str
    fitness: float  # mean cost on the training scenes; lower is better
    origin: str  # e.g. "seed 3" or "round 2"


Proposer = Callable[[Candidate, Candidate], str | None]  # parents -> child code, None if failed
Evaluator = Callable[[str], float | None]  # code -> fitness, None if invalid


def pick_parents(
    population: list[Candidate], rng: random.Random, top_k: int = 5
) -> tuple[Candidate, Candidate]:
    ranked = sorted(population, key=lambda c: c.fitness)[:top_k]
    first, second = rng.sample(ranked, 2)
    return first, second


def evolve(
    seeds: list[Candidate],
    propose: Proposer,
    evaluate: Evaluator,
    rounds: int,
    children_per_round: int,
    rng: random.Random,
    on_round: Callable[[int, list[Candidate]], None] | None = None,
) -> list[Candidate]:
    if len(seeds) < 2:
        raise ValueError("need at least 2 seed programs")
    population = list(seeds)
    for round_number in range(1, rounds + 1):
        pairs = [pick_parents(population, rng) for _ in range(children_per_round)]
        # LLM calls run in parallel; scoring stays on the main thread (time limits use signals).
        with ThreadPoolExecutor(max_workers=children_per_round) as pool:
            children = list(pool.map(lambda pair: propose(*pair), pairs))
        for code in children:
            if code is None:
                continue
            fitness = evaluate(code)
            if fitness is not None:
                population.append(Candidate(code, fitness, f"round {round_number}"))
        if on_round is not None:
            on_round(round_number, population)
    return population
