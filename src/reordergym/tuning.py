"""Pick a baseline's knobs by grid search on training scenes (never held-out ones)."""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from reordergym.config import WarehouseConfig
from reordergym.scene import Scene
from reordergym.score import score_episode
from reordergym.sim import Policy, run_episode

PolicyFactory = Callable[..., Policy]  # (warehouse, **params) -> Policy
Params = dict[str, float]


def make_grid(**axes: Sequence[float]) -> list[Params]:
    """make_grid(z=[0, 1], cap_days=[1, 2]) -> every combination as a dict."""
    names = list(axes)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*axes.values())]


def per_scene_costs(
    factory: PolicyFactory,
    params: Params,
    scenes: Sequence[Scene],
    warehouse: WarehouseConfig,
    warmup: int,
) -> list[float]:
    """Episode cost on each scene, in order. Every candidate runs on the same scenes."""
    costs: list[float] = []
    for scene in scenes:
        policy = factory(warehouse, **params)
        records = run_episode(scene, warehouse, policy, initial_on_hand=[0] * scene.n_skus)
        costs.append(score_episode(records, warehouse, warmup).total)
    return costs


def mean_score(
    factory: PolicyFactory,
    params: Params,
    scenes: Sequence[Scene],
    warehouse: WarehouseConfig,
    warmup: int,
) -> float:
    """Average episode cost (fair comparison: same scenes for every candidate)."""
    costs = per_scene_costs(factory, params, scenes, warehouse, warmup)
    return sum(costs) / len(costs)


@dataclass(frozen=True)
class TuningResult:
    params: Params
    score: float
    all_scores: list[tuple[Params, float]]


def grid_search(
    factory: PolicyFactory,
    grid: list[Params],
    scenes: Sequence[Scene],
    warehouse: WarehouseConfig,
    warmup: int,
) -> TuningResult:
    all_scores = [
        (params, mean_score(factory, params, scenes, warehouse, warmup)) for params in grid
    ]
    best_params, best_score = min(all_scores, key=lambda pair: pair[1])
    return TuningResult(params=best_params, score=best_score, all_scores=all_scores)
