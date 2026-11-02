# ReorderGym — Design

## 1. Question

> Can inventory policies written as code by an LLM hold up better than tuned textbook
> rules when demand or suppliers change in ways the policies were never tuned for?

The simulator exists to answer that fairly: every policy faces the same futures, is
tuned or searched on training scenes only, and is judged on held-out shocks.

## 2. System at a glance

```
WarehouseConfig + family + seed
        │
        ▼
generate_scene ──► Scene (demand[sku, day], lead_time[sku, day], drawn up front)
        │
        ▼
run_episode (sim.py) ◄── policy(obs) -> orders     one loop for every policy
        │
        ▼
score_episode (score.py) ──► one cost per scene, with a 5-way breakdown
        │
        ├── tuning.py   grid search for classical baselines (training scenes)
        ├── llm.py      LLM writes policy(obs) as Python, loaded with guardrails
        ├── search.py   evolutionary program search on training scenes
        └── stats.py    paired comparisons on held-out test scenes
```

One rule holds throughout: the simulator is the only place that knows the warehouse
rules. Policies only see an observation and return orders.

## 3. Scenes

A `Scene` (`scene.py`) is a pre-drawn year: `demand` and `lead_time`, both integer
arrays of shape (products, days), plus `family` and an optional `change_point`. It
checks two things only: the arrays have matching 2D shapes, and lead times are at
least 1.

### 3.1 Generator (`generate.py`)

| Part | How it is drawn |
|---|---|
| Base demand per product | Uniform 5 to 40 units/day, per scene |
| Weekly rhythm | Fixed Mon–Sun multipliers averaging 1 (weekend highest) |
| Daily demand | Poisson around base × weekday multiplier |
| Lead time | Product's nominal lead time + Poisson(0.3) delay |

`lead_time[sku, day]` is the lead time of an order **placed** on that day. Keying by
placement day means a policy's ordering cannot change the randomness it faces.

### 3.2 Families

| Family | Use | Definition |
|---|---|---|
| `stable` | Training, validation, test | The generator above, nothing else |
| `varied` | Training (alternative) | Demand drifts linearly to 0.6–1.6× by year end; every supplier is 0–3 days slower than nominal from day 0 |
| `demand_spike` | Held-out test | From a change point in the middle third of the year, demand is redrawn at 2–4× its normal level, for all products |
| `supplier_disruption` | Held-out test | From a change point in the middle third, every lead time gains 4–8 days |

Shock draws happen after the calm draws, so a shocked scene equals the stable scene
with the same seed on every day before its change point.

`varied` exists to give training some variety without ever showing the held-out
shapes: its changes are gradual or present from day 0, never sudden.

## 4. Simulator (`sim.py`)

### 4.1 Daily order of events

1. **Receive** orders arriving today.
2. **Scrap overflow.** If total stock exceeds capacity, the excess is scrapped from
   today's deliveries, split in proportion to each product's delivery (whole units,
   leftover units to the largest remainders, ties to the lower product index).
3. **Demand.** Sell `min(shelf, demand)`; the rest is lost.
4. **Holding.** Charge for stock left overnight.
5. **Order.** Call the policy; each positive order arrives on `day + lead_time`.

An order placed tonight with lead time L arrives on the morning of day L, before that
day's demand, so it must cover L days of demand, not L + 1.

Starting stock above capacity is rejected, which guarantees overflow never exceeds
today's deliveries.

### 4.2 Costs

Recorded per day, never only as a total:

| Component | Charged when |
|---|---|
| Purchase | Unit cost × quantity, when the order is placed |
| Fixed | Order fee, per product ordered that day |
| Holding | Per unit left overnight |
| Lost sales | Per unit of unmet demand |
| Scrap | Disposal fee per scrapped unit (purchase was already paid) |

The breakdown makes odd behaviour visible, such as never ordering (lost sales) or
flooding the warehouse (scrap).

### 4.3 Observation and policy interface

```python
def policy(obs: Observation) -> list[int]: ...
```

`Observation` is a TypedDict of plain lists, per product:

- `on_hand`: stock on the shelf after today's demand
- `on_order`: units ordered but not yet arrived
- `demand_history`: up to the last 56 days of demand, oldest first, including today
- `lead_time_history`: lead times of up to the last 10 deliveries, oldest first

A lead time becomes visible only when the delivery arrives, so a supplier slowdown is
noticed late. The day number, episode length and future arrays are never visible.
Policies receive copies, so they cannot change the simulator's state.

## 5. Scoring (`score.py`)

- The first 60 days are warm-up and excluded.
- The scored window "buys" the stock it inherits at the end of warm-up (on hand plus
  on order, at unit cost) and "sells back" what is left at the end at the same price.
  Inherited stock is not free and leftover stock is not wasted; only holding cost
  penalises carrying it.
- Policies never see the day number, so they cannot target the warm-up or the end.

## 6. Classical baselines

All in `policies.py`, all built on one target:

```
position = on hand + on order
target   = mean demand × L + z × std of demand × √L
```

| Baseline | Rule | Knobs |
|---|---|---|
| Base-stock | Order up to target every day; L = nominal lead time | z |
| (s,S) | If position < target, order up to target + `order_days` × mean | z, order_days |
| Capped base-stock | Base-stock, but each order ≤ `cap_days` × mean | z, cap_days |
| Adaptive base-stock | Demand stats from the last `window` days; L = mean of recent deliveries | z, window |

Each scene draws its own demand levels, so every baseline estimates mean and spread
from `demand_history`; none uses numbers fixed in advance.

Capacity has no special handling in the baselines: a target that is too high pays
scrap, and tuning learns to avoid it.

**Tuning** (`tuning.py`) is a grid search on 20 training scenes. Every candidate runs
on the same scenes. Grids: z ∈ {0, 1, 2, 3, 4, 5, 6, 8}; `order_days` ∈ {1, 2, 4, 7};
`cap_days` ∈ {0.5, 0.75, 1, 1.5, 2, 3}; `window` ∈ {7, 14, 28, 56}.

## 7. LLM policies (`llm.py`)

### 7.1 Prompt

The prompt describes the warehouse (each product's costs, max order and usual lead
time, and the capacity), the daily order of events, the goal (lowest total cost),
the observation fields, the function signature, and the rules: only `math` and
`statistics`, no global state, must be fast. It never mentions demand spikes or
supplier disruptions.

For search, `build_improve_prompt` appends two existing programs with their average
training cost and asks for a new program with lower cost.

### 7.2 Model call

Claude Opus 5.5 (`claude-opus-5-5`), effort `high`, streamed, up to 32,000 output
tokens, with server-side refusal fallback. A response that hits the token limit, is
refused or has no Python block raises `GenerationError`, which carries the cost of
the failed call so spend is counted correctly.

### 7.3 Guardrails

These are guardrails, not a security sandbox. The code runs in the same process.

- **Static checks** (AST): imports only `math` and `statistics`; no `global` or
  `nonlocal`; no attributes starting with `_` or frame prefixes (`f_`, `gi_`, ...);
  no `eval`, `exec`, `open`, `getattr` and similar names; must define `policy`.
- **Restricted builtins**: a short allowlist, plus an import hook for the two allowed
  modules.
- **Fresh namespace per episode**, so no state carries between scenes.
- **Output checks**: a list of the right length of integers within `[0, max_order]`;
  anything else, or an exception, makes the program invalid.
- **Time limit** via `SIGALRM` (Unix only), 120 s per scene set.

## 8. Program search (`search.py`, `scripts/search.py`)

- **Seeds**: zero-shot programs from an earlier run (`runs/zero_shot_20/`), scored
  on the training scenes.
- **Each round**: pick parent pairs uniformly from the 5 best programs, request
  children in parallel, score each child on the main thread (the time limit needs
  signals), and add valid children to the population.
- **Fitness**: mean cost on 10 fixed training scenes of the chosen family:
  `stable` (`--fitness calm`) or `varied` (`--fitness varied`).
- **Baselines** in the same run are tuned on 20 scenes of the same family, so both
  sides see the same training conditions.
- **Reported**: the best program by training fitness after each round, evaluated on
  the held-out test scenes. Selection never looks at test results.
- **Budget**: `--budget` (USD, default 15) stops new proposals once reached.

## 9. Experiment setup and evaluation

Shared setup lives in `experiment.py`:

| Setting | Value |
|---|---|
| Products | 8, nominal lead times 2, 3, 4, 5, 2, 3, 4, 5 days |
| Costs per product | Unit 2, holding 0.05/night, lost sale 5, order fee 10, scrap 1 |
| Max order / capacity | 1,000 per product / 600 units total |
| Horizon / warm-up | 365 days / 60 days |
| Seeds | Training from 0, validation from 10,000, test from 20,000 |
| Test set | 20 scenes each of `stable`, `demand_spike`, `supplier_disruption` |

Every policy starts each scene with zero stock.

**Statistics** (`stats.py`): for each family, each policy is compared with the best
baseline in that family, scene by scene. Reported: mean difference, a 95% percentile
bootstrap interval (10,000 resamples), and the share of scenes won. A summary
"worst-case gap" gives each policy's largest percentage gap to the best baseline
across the three families; lower means more robust.

**Scripts**:

| Script | What it runs | API key |
|---|---|---|
| `scripts/baselines.py` | Tune baselines on stable scenes, test on all families | No |
| `scripts/zero_shot.py` | Generate N programs, select on validation, test all | Yes (or `--reuse`) |
| `scripts/search.py` | Program search with calm or varied fitness | Yes |

Outputs go to `runs/`, which git ignores.

## 10. Tests

70 tests, all run with `uv run pytest`. The core ones are worked out by hand first:

- A three-day, one-product scene: stock movement, every cost component, and the score
  with a one-day warm-up.
- A two-product capacity case: 24 units against capacity 20, split 3 and 1 by the
  largest-remainder rule.
- Each baseline's order on a hand-built observation.
- Observation history: demand window length and lead times appearing only on arrival.
- Guardrails: unsafe programs rejected, invalid orders rejected, infinite loops timed
  out.
- Search loop with a fake proposer: invalid children dropped, parents from the top 5.
- Paired statistics: constant differences give a zero-width interval; ties are not
  wins.

## 11. Known limitations

- **Guardrails are not isolation.** Untrusted code runs in-process. A real process
  sandbox is required before any large-scale training loop.
- **Small evaluation.** 20 test scenes per family, one warehouse, one cost setting,
  and shocks that hit every product at once.
- **Fitness on 10 scenes.** Search can overfit its training scenes.
- **No ceiling.** Without a hindsight oracle, it is unknown how much cost is
  avoidable in each family, especially the capacity-bound demand spike.
- **Unix-only time limit.** Windows needs WSL.
- **Unused config.** `SimConfig` in `config.py` is not used; history lengths and
  warm-up are passed directly.

## 12. Planned

- **Measurement**: a hindsight oracle per scene, 50–100 test scenes per family, and a
  second warehouse setting.
- **PPO baseline**, trained on stable scenes and on varied scenes, with the same
  observation.
- **Real data**: replay real demand (for example M5). M5 records sales, not demand, and
  has no lead times, so supplier behaviour would stay synthetic.
- **Process sandbox**, then possibly GRPO on a small model.
- **"Run the warehouse" game**: a player orders weekly and is compared with the
  baselines and the best LLM program on the same scene.
