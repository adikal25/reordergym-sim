<img src="assets/icon.svg" width="64" height="64" alt="">

# ReorderGym

A small inventory simulator for testing whether LLM-written ordering policies can match
or beat the textbook rules, especially when demand or suppliers change unexpectedly.

**Run it on your laptop:** see [docs/running-locally.md](docs/running-locally.md). The
simulator and baselines need no API key.

## What's inside

- **Simulator.** Several products share one warehouse with limited space. Demand is
  random with a weekly rhythm, and supplier lead times vary. Each day: deliveries arrive,
  overflow is scrapped, customers buy, unsold stock costs money overnight, and the policy
  decides what to order. Missed sales are lost.
- **Fair comparisons.** All randomness is drawn before a run starts, so every policy
  faces exactly the same demand and delays.
- **Scene families.** Stable demand, gradual drift with slower suppliers ("varied"), and
  two held-out shocks: a sudden demand spike and a sudden supplier slowdown.
- **Classical baselines.** Base-stock, (s,S), capped base-stock and adaptive
  base-stock, tuned by grid search on training scenes only.
- **LLM policies.** An LLM writes the policy as a plain Python function. The code runs
  with guardrails: limited imports, a time limit, and checks on every order.
- **Program search.** A small evolutionary loop shows the LLM strong programs and their
  costs, asks for a better one, and keeps valid improvements.
- **Statistics.** Paired bootstrap confidence intervals and win rates against the best
  baseline in each family.

## Planned

A hindsight oracle as a cost ceiling, a PPO baseline, and a check on real demand data.
The full design is in [docs/design.md](docs/design.md).

## Repository layout

```
src/reordergym/
  config.py       product, warehouse and simulation settings
  scene.py        a pre-drawn year of demand and lead times
  generate.py     random scene generator (stable, varied, spike, disruption)
  sim.py          the daily simulation loop and cost accounting
  score.py        episode score: warm-up excluded, stock valued at both ends
  policies.py     classical baselines
  tuning.py       grid search over baseline settings
  llm.py          prompts, API calls and guardrails for LLM-written code
  search.py       evolutionary program search
  stats.py        paired bootstrap confidence intervals
  experiment.py   shared setup for the experiment scripts
scripts/
  baselines.py    tune and test the classical baselines (no API key)
  zero_shot.py    LLM-written policies vs baselines
  search.py       program search, scored on calm or varied scenes
tests/            hand-worked and property tests
```

## Development

```bash
uv sync                 # create venv and install deps
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type check
```
