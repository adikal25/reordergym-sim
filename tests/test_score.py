import pytest

from reordergym.score import score_episode
from reordergym.sim import run_episode
from tests.test_sim import SCENE, SKU, WAREHOUSE, order_up_to_12


def test_hand_traced_score_with_warmup_of_one_day() -> None:
    # Same 3-day run as test_sim. Day 0 is warm-up; days 1-2 are scored.
    #   day 1 costs: lost 10 + fixed 10 + purchase 24 = 44.0
    #   day 2 costs: hold 3.5 + fixed 10 + purchase 10 = 23.5
    #   opening stock (end of day 0): 4 on hand + 8 on order = 12 units x 2 = 24
    #   closing stock (end of day 2): 7 on hand + 5 on order = 12 units x 2 = 24
    records = run_episode(SCENE, WAREHOUSE, order_up_to_12, initial_on_hand=[10])

    score = score_episode(records, WAREHOUSE, warmup=1)

    assert score.costs.total == 44.0 + 23.5
    assert score.opening_stock_value == 12 * SKU.unit_cost
    assert score.closing_stock_value == 12 * SKU.unit_cost
    assert score.total == 67.5


def test_records_track_units_on_order() -> None:
    records = run_episode(SCENE, WAREHOUSE, order_up_to_12, initial_on_hand=[10])
    assert [r.on_order_end[0] for r in records] == [8, 12, 5]


@pytest.mark.parametrize("warmup", [0, 3])
def test_warmup_must_leave_a_scored_window(warmup: int) -> None:
    records = run_episode(SCENE, WAREHOUSE, order_up_to_12, initial_on_hand=[10])
    with pytest.raises(ValueError):
        score_episode(records, WAREHOUSE, warmup=warmup)
