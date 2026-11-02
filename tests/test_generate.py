import numpy as np

from reordergym.generate import generate_scene
from tests.test_policies import WAREHOUSE


def test_same_seed_gives_same_scene() -> None:
    a = generate_scene(WAREHOUSE, 100, "varied", seed=3)
    b = generate_scene(WAREHOUSE, 100, "varied", seed=3)
    assert np.array_equal(a.demand, b.demand) and np.array_equal(a.lead_time, b.lead_time)


def test_varied_suppliers_are_never_faster_than_nominal_and_have_no_shock() -> None:
    for seed in range(20):
        scene = generate_scene(WAREHOUSE, 100, "varied", seed=seed)
        assert scene.lead_time.min() >= WAREHOUSE.skus[0].nominal_lead_time
        assert scene.change_point is None
