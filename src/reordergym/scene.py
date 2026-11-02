"""A scene: one pre-rolled future (demand and lead times), fixed before an episode."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Family = Literal["stable", "varied", "demand_spike", "supplier_disruption"]


@dataclass(frozen=True)
class Scene:
    demand: npt.NDArray[np.int64]  # (n_skus, horizon)
    lead_time: npt.NDArray[np.int64]  # (n_skus, horizon), keyed by order day
    family: Family
    change_point: int | None = None  # day the shock starts; None if calm

    def __post_init__(self) -> None:
        if self.demand.ndim != 2 or self.demand.shape != self.lead_time.shape:
            raise ValueError("demand and lead_time must be 2D arrays of the same shape")
        if self.lead_time.min() < 1:
            raise ValueError("lead times must be >= 1")

    @property
    def n_skus(self) -> int:
        return int(self.demand.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.demand.shape[1])
