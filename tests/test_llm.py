import pytest

from reordergym.config import WarehouseConfig
from reordergym.llm import (
    ProgramError,
    build_improve_prompt,
    check_program,
    extract_code,
    load_policy,
    time_limit,
)
from reordergym.sim import Observation
from tests.test_sim import SKU

WAREHOUSE = WarehouseConfig(skus=(SKU, SKU), capacity=1000)
OBS: Observation = {
    "on_hand": [5, 5],
    "on_order": [0, 0],
    "demand_history": [[3, 4], [1, 2]],
    "lead_time_history": [[], []],
}

GOOD = """
import math

def policy(obs):
    return [max(0, 10 - h - o) for h, o in zip(obs["on_hand"], obs["on_order"])]
"""


def test_good_program_loads_and_runs() -> None:
    policy = load_policy(GOOD, WAREHOUSE)
    assert policy(OBS) == [5, 5]


def test_extract_code_takes_the_python_block() -> None:
    assert extract_code("Here you go:\n```python\nx = 1\n```\nDone.") == "x = 1\n"
    with pytest.raises(ProgramError):
        extract_code("no code here")


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef policy(obs): return [0, 0]",
        "from subprocess import run\ndef policy(obs): return [0, 0]",
        "def policy(obs): return eval('[0, 0]')",
        "def policy(obs): return obs.__class__",
        "def policy(obs):\n    g = (x for x in [1])\n    return g.gi_frame",
        "COUNT = 0\ndef policy(obs):\n    global COUNT\n    return [0, 0]",
        "def helper(obs): return [0, 0]",
        "def policy(obs) return [0, 0]",
    ],
)
def test_unsafe_or_broken_programs_are_rejected(code: str) -> None:
    with pytest.raises(ProgramError):
        check_program(code)


@pytest.mark.parametrize(
    "returns",
    ["[1]", "[1, -1]", "[1, 2.5]", "[1, 10_000]", "(1, 1)", "[True, 1]"],
)
def test_invalid_orders_are_rejected(returns: str) -> None:
    policy = load_policy(f"def policy(obs): return {returns}", WAREHOUSE)
    with pytest.raises(ProgramError):
        policy(OBS)


def test_policy_exceptions_become_program_errors() -> None:
    policy = load_policy("def policy(obs): return [1 // 0, 0]", WAREHOUSE)
    with pytest.raises(ProgramError):
        policy(OBS)


def test_infinite_loop_times_out() -> None:
    policy = load_policy("def policy(obs):\n    while True:\n        pass", WAREHOUSE)
    with pytest.raises(ProgramError), time_limit(0.2):
        policy(OBS)


def test_improve_prompt_shows_parents_and_costs() -> None:
    prompt = build_improve_prompt("BASE", [("def policy(obs): return [1]", 1234.0), (GOOD, 99.0)])
    assert prompt.startswith("BASE")
    assert "Policy A (average cost per simulated year: 1,234)" in prompt
    assert "Policy B (average cost per simulated year: 99)" in prompt
