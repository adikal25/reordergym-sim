"""Classical baselines. Each factory returns a policy: observation -> orders.

All four share one idea: keep the inventory position (shelf + on the way) near a
target that covers demand over the lead time plus safety stock:

    target = mean_demand * L + z * std_demand * sqrt(L)

An order placed tonight with lead time L arrives before demand on day L, so the
position must cover L days. Average demand is estimated from the history window,
because each scene's demand level is unknown in advance.
"""

import math
from collections.abc import Sequence

from reordergym.config import WarehouseConfig
from reordergym.sim import Observation, Policy


def mean_std(values: Sequence[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def target_level(mean: float, std: float, lead_time: float, z: float) -> float:
    return mean * lead_time + z * std * math.sqrt(lead_time)


def position(obs: Observation, sku: int) -> int:
    return obs["on_hand"][sku] + obs["on_order"][sku]


def base_stock(warehouse: WarehouseConfig, z: float) -> Policy:
    """Every day, top the position up to the target. Lead time = supplier's nominal."""
    lead_times = [sku.nominal_lead_time for sku in warehouse.skus]

    def policy(obs: Observation) -> list[int]:
        orders = []
        for sku, lead_time in enumerate(lead_times):
            mean, std = mean_std(obs["demand_history"][sku])
            target = target_level(mean, std, lead_time, z)
            orders.append(max(0, round(target - position(obs, sku))))
        return orders

    return policy


def s_S(warehouse: WarehouseConfig, z: float, order_days: float) -> Policy:
    """Wait until the position drops below s, then order up to S = s + order_days of demand.

    Fewer, bigger orders: pays the fixed order fee less often.
    """
    lead_times = [sku.nominal_lead_time for sku in warehouse.skus]

    def policy(obs: Observation) -> list[int]:
        orders = []
        for sku, lead_time in enumerate(lead_times):
            mean, std = mean_std(obs["demand_history"][sku])
            reorder_point = target_level(mean, std, lead_time, z)
            if position(obs, sku) < reorder_point:
                order_up_to = reorder_point + order_days * mean
                orders.append(max(0, round(order_up_to - position(obs, sku))))
            else:
                orders.append(0)
        return orders

    return policy


def capped_base_stock(warehouse: WarehouseConfig, z: float, cap_days: float) -> Policy:
    """Base-stock, but never order more than cap_days of average demand at once."""
    lead_times = [sku.nominal_lead_time for sku in warehouse.skus]

    def policy(obs: Observation) -> list[int]:
        orders = []
        for sku, lead_time in enumerate(lead_times):
            mean, std = mean_std(obs["demand_history"][sku])
            target = target_level(mean, std, lead_time, z)
            gap = target - position(obs, sku)
            orders.append(max(0, round(min(gap, cap_days * mean))))
        return orders

    return policy


def adaptive_base_stock(warehouse: WarehouseConfig, z: float, window: int) -> Policy:
    """Base-stock that re-estimates from recent data: demand over the last `window`
    days, lead time from recent deliveries (nominal until the first one arrives)."""
    nominal = [sku.nominal_lead_time for sku in warehouse.skus]

    def policy(obs: Observation) -> list[int]:
        orders = []
        for sku in range(len(nominal)):
            mean, std = mean_std(obs["demand_history"][sku][-window:])
            seen = obs["lead_time_history"][sku]
            lead_time = sum(seen) / len(seen) if seen else nominal[sku]
            target = target_level(mean, std, lead_time, z)
            orders.append(max(0, round(target - position(obs, sku))))
        return orders

    return policy
