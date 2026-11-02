"""Generate realistic random scenes: base level x weekly rhythm x noise, plus shocks."""

import numpy as np

from reordergym.config import WarehouseConfig
from reordergym.scene import Family, Scene

# Relative demand Mon..Sun; averages to 1 so it shapes the week without changing the mean.
WEEKLY_PATTERN = np.array([0.85, 0.75, 0.95, 0.90, 1.20, 1.35, 1.00])


def generate_scene(warehouse: WarehouseConfig, horizon: int, family: Family, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    n_skus = warehouse.n_skus

    # Calm year first. Shock draws come after, so a shocked scene matches
    # the calm scene with the same seed on every day before the change point.
    base_level = rng.uniform(5, 40, size=n_skus)  # typical units/day per SKU
    weekday = np.arange(horizon) % 7
    mean_demand = base_level[:, None] * WEEKLY_PATTERN[weekday][None, :]
    demand = rng.poisson(mean_demand)

    nominal_lead_time = np.array([sku.nominal_lead_time for sku in warehouse.skus])
    delays = rng.poisson(0.3, size=(n_skus, horizon))  # mostly 0, sometimes 1-2
    lead_time = nominal_lead_time[:, None] + delays

    if family == "stable":
        return Scene(demand=demand, lead_time=lead_time, family=family)

    change_point = int(rng.integers(horizon // 3, 2 * horizon // 3))
    if family == "demand_spike":
        multiplier = rng.uniform(2, 4)
        demand[:, change_point:] = rng.poisson(mean_demand[:, change_point:] * multiplier)
    else:  # supplier_disruption
        extra_days = int(rng.integers(4, 9))
        lead_time[:, change_point:] += extra_days

    return Scene(demand=demand, lead_time=lead_time, family=family, change_point=change_point)
