import numpy as np
import pytest

from reordergym.config import SkuParams, WarehouseConfig
from reordergym.scene import Scene
from reordergym.sim import Observation, run_episode, split_scrap

# Easy numbers for hand-checking: buy at 2, hold at 0.5/unit/night,
# lose 5 per missed sale, pay 10 per order placed.
SKU = SkuParams(
    unit_cost=2.0,
    holding_cost=0.5,
    lost_sales_penalty=5.0,
    fixed_order_cost=10.0,
    scrap_cost=1.0,
    max_order=100,
    nominal_lead_time=1,
)
WAREHOUSE = WarehouseConfig(skus=(SKU,), capacity=1000)

# 1 SKU, lead time 1, demand 6 / 14 / 5.
SCENE = Scene(demand=np.array([[6, 14, 5]]), lead_time=np.array([[1, 1, 1]]), family="stable")


def order_up_to_12(obs: Observation) -> list[int]:
    return [
        max(0, 12 - on_hand - on_order)
        for on_hand, on_order in zip(obs["on_hand"], obs["on_order"], strict=True)
    ]


def test_hand_traced_stock_movement() -> None:
    records = run_episode(SCENE, WAREHOUSE, order_up_to_12, initial_on_hand=[10])

    expected = [
        # arrived, sold, lost, on_hand_end, ordered
        (0, 6, 0, 4, 8),
        (8, 12, 2, 0, 12),
        (12, 5, 0, 7, 5),
    ]
    actual = [(r.arrived[0], r.sold[0], r.lost[0], r.on_hand_end[0], r.ordered[0]) for r in records]
    assert actual == expected


def test_hand_traced_costs() -> None:
    records = run_episode(SCENE, WAREHOUSE, order_up_to_12, initial_on_hand=[10])

    expected = [
        # holding     lost sales  fixed  purchase
        (4 * 0.5, 0 * 5.0, 10.0, 8 * 2.0),  # day 0: total 28.0
        (0 * 0.5, 2 * 5.0, 10.0, 12 * 2.0),  # day 1: total 44.0
        (7 * 0.5, 0 * 5.0, 10.0, 5 * 2.0),  # day 2: total 23.5
    ]
    actual = [(r.cost.holding, r.cost.lost_sales, r.cost.fixed, r.cost.purchase) for r in records]
    assert actual == expected
    assert sum(r.cost.total for r in records) == 95.5


def test_no_order_means_no_order_cost() -> None:
    records = run_episode(SCENE, WAREHOUSE, lambda obs: [0], initial_on_hand=[10])
    assert all(r.cost.fixed == 0 and r.cost.purchase == 0 for r in records)


def test_sku_count_mismatch_is_rejected() -> None:
    two_skus = WarehouseConfig(skus=(SKU, SKU), capacity=1000)
    with pytest.raises(ValueError):
        run_episode(SCENE, two_skus, order_up_to_12, initial_on_hand=[10])


def test_hand_traced_capacity_and_scrap() -> None:
    # 2 SKUs share capacity 20. Start with 8 + 4 = 12, sell 3 each on day 0 -> 5 + 1.
    # Day 0 order 12 + 6 arrives on day 1: 17 + 7 = 24, so 4 units must be scrapped.
    # Split 4 by today's deliveries 12 : 6 -> exact shares 2.67 and 1.33
    # -> whole parts 2 and 1, leftover unit to the larger remainder (SKU 0) -> 3 and 1.
    warehouse = WarehouseConfig(skus=(SKU, SKU), capacity=20)
    scene = Scene(
        demand=np.array([[3, 3], [3, 3]]), lead_time=np.array([[1, 1], [1, 1]]), family="stable"
    )
    plan = iter([[12, 6], [0, 0]])

    records = run_episode(scene, warehouse, lambda obs: next(plan), initial_on_hand=[8, 4])

    day1 = records[1]
    assert day1.arrived == [12, 6]
    assert day1.scrapped == [3, 1]
    assert day1.on_hand_end == [14 - 3, 6 - 3]  # 20 after scrap, then demand of 3 each
    assert day1.cost.scrap == 4 * SKU.scrap_cost
    assert records[0].scrapped == [0, 0]


def test_split_scrap_breaks_ties_by_sku_index() -> None:
    assert split_scrap(1, [5, 5]) == [1, 0]
    assert split_scrap(3, [1, 1, 1]) == [1, 1, 1]


def test_initial_stock_over_capacity_is_rejected() -> None:
    small = WarehouseConfig(skus=(SKU,), capacity=5)
    with pytest.raises(ValueError):
        run_episode(SCENE, small, order_up_to_12, initial_on_hand=[10])


def test_observation_history_by_hand() -> None:
    # Orders of 1 every day. Day 0's order takes 2 days, later ones 1 day,
    # so both day-0 and day-1 orders land on day 2.
    scene = Scene(
        demand=np.array([[1, 2, 3, 4]]), lead_time=np.array([[2, 1, 1, 1]]), family="stable"
    )
    seen: list[Observation] = []

    def recording_policy(obs: Observation) -> list[int]:
        seen.append(obs)
        return [1]

    run_episode(scene, WAREHOUSE, recording_policy, initial_on_hand=[10], demand_history_days=2)

    assert [obs["demand_history"][0] for obs in seen] == [[1], [1, 2], [2, 3], [3, 4]]
    assert [obs["lead_time_history"][0] for obs in seen] == [[], [], [2, 1], [2, 1, 1]]
