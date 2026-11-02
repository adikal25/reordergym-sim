from typing import Any

import pytest
from pydantic import ValidationError

from reordergym.config import SimConfig, SkuParams, WarehouseConfig


def make_sku(**overrides: Any) -> SkuParams:
    params: dict[str, Any] = {
        "unit_cost": 10.0,
        "holding_cost": 0.1,
        "lost_sales_penalty": 20.0,
        "fixed_order_cost": 5.0,
        "scrap_cost": 2.0,
        "max_order": 200,
        "nominal_lead_time": 3,
    }
    params.update(overrides)
    return SkuParams(**params)


def test_valid_warehouse_builds() -> None:
    warehouse = WarehouseConfig(skus=(make_sku(), make_sku()), capacity=500)
    assert warehouse.n_skus == 2


def test_sim_config_defaults_match_design() -> None:
    config = SimConfig()
    assert (config.horizon, config.warmup) == (365, 60)
    assert (config.demand_history, config.lead_time_history) == (56, 10)


@pytest.mark.parametrize("lead_time", [0, -1])
def test_lead_time_below_one_is_rejected(lead_time: int) -> None:
    with pytest.raises(ValidationError):
        make_sku(nominal_lead_time=lead_time)


@pytest.mark.parametrize(
    "field, value",
    [
        ("unit_cost", 0.0),
        ("holding_cost", -0.1),
        ("lost_sales_penalty", -1.0),
        ("fixed_order_cost", -1.0),
        ("scrap_cost", -1.0),
        ("max_order", 0),
    ],
)
def test_invalid_sku_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_sku(**{field: value})


def test_empty_sku_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WarehouseConfig(skus=(), capacity=500)


def test_non_positive_capacity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WarehouseConfig(skus=(make_sku(),), capacity=0)


@pytest.mark.parametrize("warmup", [365, 400])
def test_warmup_must_leave_scored_days(warmup: int) -> None:
    with pytest.raises(ValidationError):
        SimConfig(horizon=365, warmup=warmup)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_sku(typo_field=1)


def test_configs_are_immutable() -> None:
    sku = make_sku()
    with pytest.raises(ValidationError):
        sku.unit_cost = 1.0  # type: ignore[misc]


def test_warehouse_round_trips_through_json() -> None:
    warehouse = WarehouseConfig(skus=(make_sku(), make_sku(max_order=50)), capacity=500)
    restored = WarehouseConfig.model_validate_json(warehouse.model_dump_json())
    assert restored == warehouse
