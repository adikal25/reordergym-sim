"""Score an episode: sum costs over the scored window, valuing stock at both ends."""

from dataclasses import dataclass

from reordergym.config import WarehouseConfig
from reordergym.sim import Costs, DayRecord


@dataclass(frozen=True)
class EpisodeScore:
    costs: Costs  # summed over the scored days
    opening_stock_value: float  # stock inherited from warm-up, "bought" at unit cost
    closing_stock_value: float  # stock left at the end, "sold back" at unit cost

    @property
    def total(self) -> float:
        return self.costs.total + self.opening_stock_value - self.closing_stock_value


def stock_value(record: DayRecord, warehouse: WarehouseConfig) -> float:
    return sum(
        (on_hand + on_order) * sku.unit_cost
        for on_hand, on_order, sku in zip(
            record.on_hand_end, record.on_order_end, warehouse.skus, strict=True
        )
    )


def score_episode(
    records: list[DayRecord], warehouse: WarehouseConfig, warmup: int
) -> EpisodeScore:
    if not 1 <= warmup < len(records):
        raise ValueError(f"warmup must be in [1, {len(records)}), got {warmup}")
    scored = records[warmup:]
    costs = Costs(
        holding=sum(r.cost.holding for r in scored),
        lost_sales=sum(r.cost.lost_sales for r in scored),
        fixed=sum(r.cost.fixed for r in scored),
        purchase=sum(r.cost.purchase for r in scored),
        scrap=sum(r.cost.scrap for r in scored),
    )
    return EpisodeScore(
        costs=costs,
        opening_stock_value=stock_value(records[warmup - 1], warehouse),
        closing_stock_value=stock_value(records[-1], warehouse),
    )
