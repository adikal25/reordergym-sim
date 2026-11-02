<img src="assets/icon.svg" width="64" height="64" alt="">

# ReorderGym

LLMs that write inventory policies as code, trained against a verifier.

> Research question: on supply-chain shocks the model never trained on, do inventory
> policies written as code by an LLM generalize better than a neural RL policy or the
> textbook formula?

Policies are compared on a multi-SKU warehouse simulator with shared capacity,
lead-time jitter, lost sales, fixed order costs and scrap. Baselines include tuned
base-stock, (s,S), adaptive base-stock, a hindsight oracle and PPO. LLM policies come
from zero-shot prompting, FunSearch-style program search, and a small model trained
with SFT + GRPO. Rankings are checked against real demand traces from M5.

## Status

Early design. See [`docs/design.md`](docs/design.md).

## Development

```bash
uv sync                 # create venv and install deps
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type check
```
