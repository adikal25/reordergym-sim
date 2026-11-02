"""Ask an LLM to write an inventory policy as Python code, then load it safely.

Safety here is guardrails, not a sandbox: static checks reject imports, dunder and
frame access, and dangerous builtins; the code runs with a restricted set of builtins
and a time limit; its output is validated. A real process sandbox comes before any
training loop (GRPO) runs thousands of untrusted programs.
"""

import ast
import builtins
import math
import re
import signal
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Any

import anthropic

from reordergym.config import WarehouseConfig
from reordergym.sim import Observation, Policy

MODEL = "claude-opus-5-5"
PRICE_PER_INPUT_TOKEN = 4.00 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 20.00 / 1_000_000


class ProgramError(Exception):
    """The program is unsafe, broken, too slow, or returned invalid orders."""


# --- Prompt ---------------------------------------------------------------------------


def build_prompt(
    warehouse: WarehouseConfig, demand_history_days: int, lead_time_history_len: int
) -> str:
    sku_lines = "\n".join(
        f"  SKU {i}: unit cost {s.unit_cost}, holding {s.holding_cost}/unit/night, "
        f"lost sale {s.lost_sales_penalty}/unit, order fee {s.fixed_order_cost}/order, "
        f"scrap fee {s.scrap_cost}/unit, max order {s.max_order}, "
        f"usual lead time {s.nominal_lead_time} days"
        for i, s in enumerate(warehouse.skus)
    )
    return f"""Write an inventory replenishment policy for a simulated warehouse,
as a Python function.

## The warehouse
{warehouse.n_skus} SKUs share one warehouse with capacity {warehouse.capacity} units in total.
{sku_lines}

## What happens each day, in order
1. Receive: orders due today arrive.
2. Capacity: if total stock exceeds capacity, the excess is scrapped from today's
   deliveries (scrap fee per unit; the purchase cost is already paid).
3. Demand: customers buy what is on the shelf. Unmet demand is lost (lost-sale cost).
4. Holding: stock left overnight costs the holding cost.
5. Your policy is called and returns one order quantity per SKU. Each order > 0 costs
   the order fee plus unit cost x quantity, and arrives on the morning of day + lead time,
   before that day's demand.

Demand and lead times are random and not known in advance. Episodes last many months.
The goal is the lowest total cost: purchases + holding + lost sales + order fees + scrap.

## Your function
```python
def policy(obs: dict) -> list[int]:
    ...
```
`obs` contains (index i = SKU i):
- obs["on_hand"][i]: int, units on the shelf now (after today's demand)
- obs["on_order"][i]: int, units ordered but not yet arrived
- obs["demand_history"][i]: list[int], daily demand for up to the last
  {demand_history_days} days, oldest first, including today (shorter early on)
- obs["lead_time_history"][i]: list[int], actual lead times of up to the last
  {lead_time_history_len} deliveries received, oldest first (a delivery's lead
  time is only known once it arrives; empty early on)

Return a list of {warehouse.n_skus} non-negative ints, each at most that SKU's max order.

## Rules
- Pure Python. You may import only `math` and `statistics`.
- No global state: the function must depend only on `obs`.
- It is called once per simulated day and must be fast.

Reply with the complete code in one ```python block."""


# --- Calling the model ----------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedProgram:
    code: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_INPUT_TOKEN + self.output_tokens * PRICE_PER_OUTPUT_TOKEN
        )


def extract_code(text: str) -> str:
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        raise ProgramError("response has no ```python block")
    return str(blocks[-1])


def write_policy(client: anthropic.Anthropic, prompt: str) -> GeneratedProgram:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise ProgramError("model declined the request")
    if response.stop_reason == "max_tokens":
        raise ProgramError("response hit max_tokens before finishing")
    text = "".join(block.text for block in response.content if block.type == "text")
    return GeneratedProgram(
        code=extract_code(text),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


# --- Loading the program safely -------------------------------------------------------

ALLOWED_IMPORTS = {"math": math, "statistics": statistics}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "exit", "quit", "help", "dir", "type", "object",
    "super", "memoryview", "classmethod", "staticmethod", "property",
}  # fmt: skip
FRAME_ATTR_PREFIXES = ("_", "f_", "gi_", "cr_", "ag_", "tb_")
SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "ValueError", "ZeroDivisionError", "Exception",
]  # fmt: skip


def check_program(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ProgramError(f"syntax error: {error}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module not in ALLOWED_IMPORTS:
                raise ProgramError(f"import of {module!r} is not allowed")
        if isinstance(node, ast.Global | ast.Nonlocal):
            raise ProgramError("global state is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith(FRAME_ATTR_PREFIXES):
            raise ProgramError(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ProgramError(f"name {node.id!r} is not allowed")
    defines_policy = any(
        isinstance(node, ast.FunctionDef) and node.name == "policy" for node in tree.body
    )
    if not defines_policy:
        raise ProgramError("no top-level `def policy` found")


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return ALLOWED_IMPORTS[name]


def load_policy(code: str, warehouse: WarehouseConfig) -> Policy:
    """Check the code, run it in a fresh namespace, and wrap `policy` with output checks.

    Call once per episode: a fresh namespace means no state carries across episodes.
    """
    check_program(code)
    safe_builtins: dict[str, Any] = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe_builtins["__import__"] = _restricted_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(compile(code, "<llm-policy>", "exec"), namespace)
    except Exception as error:
        raise ProgramError(f"error while loading: {error!r}") from error
    raw_policy = namespace["policy"]
    max_orders = [sku.max_order for sku in warehouse.skus]

    def checked_policy(obs: Observation) -> list[int]:
        try:
            orders = raw_policy(obs)
        except Exception as error:
            raise ProgramError(f"policy raised {error!r}") from error
        if not isinstance(orders, list) or len(orders) != len(max_orders):
            raise ProgramError(f"expected a list of {len(max_orders)} orders, got {orders!r}")
        for qty, cap in zip(orders, max_orders, strict=True):
            if type(qty) is not int or not 0 <= qty <= cap:
                raise ProgramError(f"invalid order quantity {qty!r} (must be int in [0, {cap}])")
        return orders

    return checked_policy


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise ProgramError if the block runs longer than `seconds` (main thread, Unix only)."""

    def on_timeout(signum: int, frame: FrameType | None) -> None:
        raise ProgramError(f"timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
