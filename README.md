<img src="assets/icon.svg" width="64" height="64" alt="">

# FabDispatch

A simulation-based reinforcement-learning system that decides which manufacturing job a
machine should run next. A Maskable PPO policy is benchmarked against strong dispatching
rules (FIFO, SPT, EDD, ATCS) in a simplified, semiconductor-inspired job shop with
re-entrant routing, stochastic processing times, machine failures, and setups.

> Research question: can an RL dispatching policy reduce weighted tardiness and flow
> time versus strong rules — and under which conditions does it help or hurt?

This is a simulator-backed research project, not a model of any real fab.

## Status

Early design. See [`docs/design.md`](docs/design.md) for the end-to-end design.

## Development

```bash
uv sync                 # create venv and install deps
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type check
```
