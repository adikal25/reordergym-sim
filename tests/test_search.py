import random

import pytest

from reordergym.search import Candidate, evolve, pick_parents

# Fake world: a program's "code" is just its cost written as text.


def fake_evaluate(code: str) -> float | None:
    return None if code == "broken" else float(code)


def test_children_join_the_population_and_invalid_ones_are_dropped() -> None:
    seeds = [Candidate("100", 100.0, "seed 0"), Candidate("90", 90.0, "seed 1")]
    proposals = iter(["80", "broken", "70", "75"])

    def propose(a: Candidate, b: Candidate) -> str | None:
        return next(proposals)

    population = evolve(
        seeds, propose, fake_evaluate, rounds=2, children_per_round=2, rng=random.Random(0)
    )

    assert sorted(c.fitness for c in population) == [70.0, 75.0, 80.0, 90.0, 100.0]
    assert {c.origin for c in population if c.fitness in (70.0, 75.0)} == {"round 2"}


def test_parents_come_from_the_best_programs() -> None:
    population = [Candidate(str(cost), float(cost), "seed") for cost in range(1, 21)]
    for seed in range(50):
        a, b = pick_parents(population, random.Random(seed), top_k=5)
        assert a.fitness <= 5 and b.fitness <= 5 and a is not b


def test_needs_two_seeds() -> None:
    with pytest.raises(ValueError):
        evolve(
            [Candidate("1", 1.0, "seed")], lambda a, b: "1", fake_evaluate, 1, 1, random.Random(0)
        )
