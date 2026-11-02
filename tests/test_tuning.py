from reordergym.generate import generate_scene
from reordergym.policies import base_stock
from reordergym.tuning import grid_search, make_grid, mean_score
from tests.test_sim import WAREHOUSE


def test_make_grid_lists_every_combination() -> None:
    assert make_grid(z=[0, 1], cap_days=[2, 3]) == [
        {"z": 0, "cap_days": 2},
        {"z": 0, "cap_days": 3},
        {"z": 1, "cap_days": 2},
        {"z": 1, "cap_days": 3},
    ]


def test_grid_search_returns_the_cheapest_candidate() -> None:
    scenes = [generate_scene(WAREHOUSE, horizon=120, family="stable", seed=s) for s in range(3)]
    grid = make_grid(z=[0.0, 1.0, 2.0])

    result = grid_search(base_stock, grid, scenes, WAREHOUSE, warmup=30)

    expected = [mean_score(base_stock, p, scenes, WAREHOUSE, warmup=30) for p in grid]
    assert result.score == min(expected)
    assert [score for _, score in result.all_scores] == expected
