import math

from reordergym.config import WarehouseConfig
from reordergym.policies import (
    adaptive_base_stock,
    base_stock,
    capped_base_stock,
    mean_std,
    s_S,
    target_level,
)
from reordergym.sim import Observation
from tests.test_sim import SKU

# One SKU whose supplier nominally takes 3 days.
WAREHOUSE = WarehouseConfig(skus=(SKU.model_copy(update={"nominal_lead_time": 3}),), capacity=1000)


def make_obs(
    on_hand: int, on_order: int, demand: list[int], lead_times: list[int] | None = None
) -> Observation:
    return {
        "on_hand": [on_hand],
        "on_order": [on_order],
        "demand_history": [demand],
        "lead_time_history": [lead_times or []],
    }


def test_mean_std_and_target_by_hand() -> None:
    assert mean_std([10, 30]) == (20.0, 10.0)
    assert mean_std([]) == (0.0, 0.0)
    # 20/day over 4 days = 80, plus z=1 x std 10 x sqrt(4) = 20 -> 100
    assert math.isclose(target_level(mean=20, std=10, lead_time=4, z=1), 100)


def test_base_stock_tops_up_to_target() -> None:
    # Steady 20/day, no spread, lead time 3 -> target 60. Position 10 + 20 = 30 -> order 30.
    policy = base_stock(WAREHOUSE, z=1.0)
    assert policy(make_obs(on_hand=10, on_order=20, demand=[20] * 10)) == [30]


def test_s_S_waits_until_below_reorder_point() -> None:
    # Reorder point 60 (z=0). Order up to S = 60 + 2 days x 20 = 100.
    policy = s_S(WAREHOUSE, z=0.0, order_days=2)
    assert policy(make_obs(on_hand=70, on_order=0, demand=[20] * 10)) == [0]
    assert policy(make_obs(on_hand=50, on_order=0, demand=[20] * 10)) == [50]


def test_capped_base_stock_never_orders_more_than_cap() -> None:
    # Gap to target 60 is 50, but cap is 1 day of demand = 20.
    policy = capped_base_stock(WAREHOUSE, z=0.0, cap_days=1)
    assert policy(make_obs(on_hand=10, on_order=0, demand=[20] * 10)) == [20]


def test_adaptive_learns_lead_time_from_deliveries() -> None:
    # Deliveries have taken 5 days, not the nominal 3 -> target 20 x 5 = 100.
    policy = adaptive_base_stock(WAREHOUSE, z=0.0, window=14)
    assert policy(make_obs(on_hand=30, on_order=0, demand=[20] * 10, lead_times=[5, 5])) == [70]
    # With no deliveries seen yet it falls back to nominal 3 -> target 60.
    assert policy(make_obs(on_hand=30, on_order=0, demand=[20] * 10)) == [30]


def test_adaptive_only_uses_recent_window() -> None:
    # Demand jumped from 10 to 30 a week ago. A 7-day window sees only 30s -> target 90.
    policy = adaptive_base_stock(WAREHOUSE, z=0.0, window=7)
    assert policy(make_obs(on_hand=0, on_order=0, demand=[10] * 50 + [30] * 7)) == [90]
