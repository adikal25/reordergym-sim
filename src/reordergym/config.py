"""Validated, immutable configuration for the warehouse and the simulator.

Units: one time step is one day; quantities are integer units; costs are in an
arbitrary currency unit. See docs/design.md section 2 for how each field is used.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    """Base for config models: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SkuParams(_FrozenModel):
    """Cost and ordering parameters for one SKU."""

    unit_cost: float = Field(gt=0, description="Purchase cost per unit ordered.")
    holding_cost: float = Field(ge=0, description="Cost per unit held at end of day.")
    lost_sales_penalty: float = Field(ge=0, description="Cost per unit of unmet demand.")
    fixed_order_cost: float = Field(ge=0, description="Cost per order placed (q > 0).")
    scrap_cost: float = Field(ge=0, description="Disposal cost per unit scrapped.")
    max_order: int = Field(gt=0, description="Largest allowed order quantity.")
    nominal_lead_time: int = Field(
        ge=1,
        description=(
            "Typical days from placing an order to receiving it. An order placed on "
            "day t with lead time L is received on day t + L before that day's demand, "
            "so the minimum meaningful lead time is 1 (next-morning delivery)."
        ),
    )


class WarehouseConfig(_FrozenModel):
    """SKUs sharing one warehouse with a total on-hand capacity."""

    skus: tuple[SkuParams, ...] = Field(min_length=1)
    capacity: int = Field(gt=0, description="Max total on-hand units across all SKUs.")

    @property
    def n_skus(self) -> int:
        return len(self.skus)


class SimConfig(_FrozenModel):
    """Episode length, scoring window and observation history sizes."""

    horizon: int = Field(default=365, gt=0, description="Days per episode (T).")
    warmup: int = Field(default=60, ge=0, description="Initial days excluded from scoring (W).")
    demand_history: int = Field(
        default=56, ge=1, description="Days of past demand shown to the policy (H)."
    )
    lead_time_history: int = Field(
        default=10, ge=1, description="Realised lead times of recent receipts shown (M)."
    )

    @model_validator(mode="after")
    def _warmup_leaves_scored_days(self) -> Self:
        if self.warmup >= self.horizon:
            raise ValueError(f"warmup ({self.warmup}) must be less than horizon ({self.horizon})")
        return self
