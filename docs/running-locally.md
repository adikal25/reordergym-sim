# Running ReorderGym on your own laptop

Everything runs locally on a normal laptop. The simulator and the classical baselines
need no API key. Only the LLM experiments call the Anthropic API.

## 1. Install

You need `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
`uv` installs the right Python version (3.12) for you.

```bash
git clone https://github.com/adikal25/fabdispatch.git
cd fabdispatch
uv sync
```

Check that everything works:

```bash
uv run pytest
```

Supported: macOS and Linux. On Windows, use WSL. The LLM experiments use a Unix-only
timer to stop slow programs.

## 2. Run the classical baselines (free, about 1 minute)

```bash
uv run python scripts/baselines.py
```

This tunes four textbook policies (base-stock, (s,S), capped base-stock, adaptive
base-stock) on 20 calm scenes, then tests them on 20 unseen scenes from each of three
families: stable demand, a sudden demand spike, and a supplier slowdown. It prints the
average cost per family (lower is better), paired comparisons with 95% confidence
intervals, and each policy's worst-case gap to the best baseline.

## 3. Simulate one year yourself

Paste this into `uv run python`:

```python
from reordergym.experiment import WAREHOUSE, WARMUP
from reordergym.generate import generate_scene
from reordergym.policies import base_stock
from reordergym.score import score_episode
from reordergym.sim import run_episode

scene = generate_scene(WAREHOUSE, horizon=365, family="supplier_disruption", seed=7)
policy = base_stock(WAREHOUSE, z=3.0)
records = run_episode(scene, WAREHOUSE, policy, initial_on_hand=[0] * scene.n_skus)
score = score_episode(records, WAREHOUSE, warmup=WARMUP)

print(f"supplier slows down on day {scene.change_point}")
print(f"total cost: {score.total:,.0f}")
print(score.costs)  # holding, lost sales, order fees, purchases, scrap
```

Things to try:

- **Change the scene.** `family` can be `"stable"`, `"varied"`, `"demand_spike"` or
  `"supplier_disruption"`. Change `seed` for a different random year.
- **Change the policy.** Try `adaptive_base_stock(WAREHOUSE, z=3.0, window=14)` or
  `capped_base_stock(WAREHOUSE, z=5.0, cap_days=1)` from `reordergym.policies`.
- **Write your own policy.** Any function that takes the observation and returns one
  order per product works:

```python
def order_up_to_100(obs):
    return [
        max(0, 100 - on_hand - on_order)
        for on_hand, on_order in zip(obs["on_hand"], obs["on_order"])
    ]


records = run_episode(scene, WAREHOUSE, order_up_to_100, initial_on_hand=[0] * scene.n_skus)
print(f"{score_episode(records, WAREHOUSE, warmup=WARMUP).total:,.0f}")
```

The observation contains, per product: `on_hand`, `on_order`, `demand_history` (recent
daily demand) and `lead_time_history` (recent actual delivery times). The day number and
the future are never visible.

## 4. LLM experiments (optional, costs money)

Put your Anthropic API key in a `.env` file in the project folder. Git ignores it.

```
ANTHROPIC_API_KEY=sk-ant-...
```

**Zero-shot programs.** The LLM writes policies from the task description alone, and
each one is tested like the baselines.

```bash
uv run --env-file .env python scripts/zero_shot.py --n-programs 4
```

Four programs cost roughly $0.50 to $1. Programs and results are saved in
`runs/zero_shot/`. Rerun the scoring without new API calls:

```bash
uv run python scripts/zero_shot.py --reuse
```

**Program search.** Starting from saved zero-shot programs, the LLM repeatedly sees
strong programs with their costs and writes improved ones. Generate the starting
programs first, then search:

```bash
uv run --env-file .env python scripts/zero_shot.py --n-programs 20 --out runs/zero_shot_20
uv run --env-file .env python -u scripts/search.py --fitness calm --seed 0
```

`--fitness varied` scores programs on scenes with gradual demand drift and slower
suppliers instead of calm scenes. `--rounds` and `--children` set the search size, and
`--budget` caps the API spend in dollars (default 15). A 6-round, 6-child search cost
roughly $9 to $12 in our runs. Results go to `runs/search_<fitness>_seed<seed>/`.

Costs are estimates. The scripts print what each run actually spent.
