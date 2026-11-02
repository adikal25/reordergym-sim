# ReorderGym — End-to-End Design

Status: draft for review. Nothing here is implemented yet.
Supersedes the FabDispatch design (recoverable from git commit `1cc089a`).

## 0. Research question

> On supply-chain shocks the model never trained on, do inventory policies written
> as code by an LLM generalize better than a neural RL policy or the textbook formula?

Success = a trustworthy, paired, held-out comparison with confidence intervals and
readable winning programs. A negative result ("LLM programs do not transfer") is a
valid outcome and gets reported the same way.

---

## 1. System at a glance

```
ScenarioSpec (Pydantic) + seed
        │
        ▼
SceneGenerator ──► Scene (ALL randomness pre-sampled: demand[i,t], lead_time[i,t])
        │                 frozen eval scenes are hashed into a manifest
        ▼
Fast sim (trusted process)  ◄── one core, shared by every policy
   obs_t ──► policy(obs) ──► orders_t
        │
   ┌────┼──────────────┬───────────────────┬─────────────────────┐
   ▼    ▼              ▼                   ▼                     ▼
Classical    PPO (SB3)     Verifier (sandboxed      LLM generators:        Game
baselines                  untrusted programs)      zero-shot, program     (Streamlit)
+ oracle                        │                   search, SFT+GRPO
   │                            └── reward ─────────────┘
   ▼
Evaluator: paired CRN scenes, train / in-dist test / held-out families, CIs
   │
   ├──► Sim results table (% gap closed)
   └──► Real-trace judge (M5 demand replay) → sim-vs-real ranking agreement
```

Structural rule (same as FactorioGym): **one sim core, one obs schema, one cost
function**. Every policy — formula, network, program, human player — sees the same
observation and is charged by the same code. Only the decision differs.

---

## 2. Simulator

### 2.1 Entities and time

Periodic review, one step = one day, horizon `T` (default 365), `N` SKUs (default 8)
sharing one warehouse of capacity `C` units. Integer units.

Per SKU `i`: unit cost `c_i`, holding cost `h_i` per unit-day, lost-sales penalty
`p_i` per unit, fixed order cost `K_i` per order placed, scrap cost `x_i` per unit,
max order `q_max_i`.

### 2.2 Daily event order (fixed, documented, tested)

1. **Receive** every outstanding order whose arrival day is `t`. Orders may cross.
2. **Capacity check**: if total on-hand > `C`, the excess is scrapped *from today's
   receipts*, split across SKUs proportionally to today's receipt quantities
   (largest-remainder rounding, ties by SKU index). Charged `x_i` per unit; the
   purchase cost `c_i` was already paid when the order was placed.
3. **Demand** `D[i,t]` is served from on-hand. Unmet demand is lost, charged `p_i`.
4. **Holding** charged on end-of-day on-hand: `h_i · on_hand_i`.
5. **Observe → decide**: the policy returns integer orders `q_i ∈ [0, q_max_i]`.
   Each `q_i > 0` costs `K_i + c_i · q_i` and arrives on day `t + L[i,t]`.
   Lead times are ≥ 1. An order placed tonight with lead time L is received in
   step 1 of day `t + L`, *before* that day's demand, so it must cover L days of
   demand (not L + 1). L = 1 is next-morning delivery.

The capacity semantics above are exactly where the prototype's base-stock bug came
from. They are written once, in the sim, and baselines must be capacity-aware
against *this* definition (§5.1). A hand-traced test pins them down.

### 2.3 End effects (anti-reward-hacking by construction)

- **Policies never see `t` or `T`.** They see day-of-week only (needed for
  seasonality). A policy cannot "stop ordering near the end" if it cannot tell
  where the end is.
- **Warm-up**: costs in days `[0, W)` (default W = 60) are excluded from scoring.
  Initial state: on-hand = `μ_i · (L̄_i + 1)`, empty pipeline.
- **Stock valuation at both ends of the scored window**: the window is charged the
  stock it inherits from warm-up (on-hand + pipeline at end of day W−1) at `c_i`, and
  credited the stock left at `T` at `c_i`. Inherited stock is not free, leftover
  stock is not wasted, and only holding cost penalises carrying it.

### 2.4 Cost accounting

Total cost is the sum of five components, always reported separately:
`purchase + holding + lost_sales + fixed + scrap` (minus terminal credit).
Breakdowns are what expose hacks such as "never order, eat lost sales" or
"flood the warehouse, eat scrap".

---

## 3. Randomness and common random numbers

Every scene pre-samples, before simulation starts:

| Array | Meaning |
|---|---|
| `demand[i, t]` | demand for SKU i on day t |
| `lead_time[i, t]` | lead time of an order for SKU i **placed on day t** |

Keying lead time by *placement day* (not "the next order") means a policy's order
timing cannot change the randomness it faces. Every policy faces identical draws on
a scene. Streams come from `numpy.random.SeedSequence(seed).spawn(...)`.

The sim is deterministic given a scene: same scene + same policy ⇒ identical
trajectory (a test).

---

## 4. Observation and policy interface

### 4.1 Observation (no future leakage)

Plain Python data (dicts, lists, ints, floats) so it can cross a process boundary:

- **Static**: `n_skus`, `capacity`, per-SKU `c, h, p, K, x, q_max`, nominal lead time.
- **Per SKU**: `on_hand`, `pipeline` as `(qty, days_since_placed)` pairs (arrival
  day is *not* revealed), `demand_history` (last H = 56 days), `lead_time_history`
  (realised lead times of the last M = 10 received orders).
- **Global**: `day_of_week`, total on-hand.

Never observed: `t`, `T`, future demand, future lead times, the scene family.

`lead_time_history` and `demand_history` are what make shock *detection* possible
("orders are arriving late" / "demand stepped up").

### 4.2 Policy interface

```python
def policy(obs: dict) -> list[int]: ...
```

**Stateless.** Memory lives in the obs history windows. This keeps programs pure,
deterministic, replayable, and easy to sandbox. Any history a policy needs must fit
the window — a stated limitation.

Output validation is strict: wrong length, non-int, negative, or > `q_max` makes the
episode **invalid** (not silently clipped). Validity rate is a reported metric.

---

## 5. Policies compared

All policies are tuned / trained / searched on the **train family only**, with
hyperparameter selection on train-family **validation** scenes. Nobody touches
held-out families before final evaluation.

| Policy | Role | Notes |
|---|---|---|
| Static base-stock | Textbook floor | Order-up-to `S_i`, capacity-aware scaling; safety factor tuned hard on train |
| Static (s,S) | Textbook floor | Handles fixed cost `K`; tuned on train |
| Capped base-stock | Strong lost-sales heuristic | Order-up-to with a per-order cap (Xin 2021); the baseline recent LLM-inventory work (InvEvolve) compares against |
| **Adaptive base-stock** | Strong classical | Rolling estimates of μ, σ, L from the obs windows, recomputed daily |
| Hindsight oracle | Ceiling (heuristic) | Per-scene, see §5.2 |
| PPO | Neural RL | §6 |
| Zero-shot LLM | Prior knowledge | Frontier API model + the small base model |
| Program search | LLM evolution | FunSearch-style, §7.2 |
| Small LLM, SFT + GRPO | Main contribution | §7.3 |

### 5.1 Why adaptive base-stock is mandatory

The headline claim is "programs detect shocks and adapt". The obvious reviewer
objection is "so would a textbook policy that re-estimates demand and lead time on
a rolling window". Without this baseline a win on held-out shocks is not credible.
"Floor" in every table = the best of the four classical policies on train
validation.

### 5.2 Oracle and "% gap closed"

```
gap_closed(π) = (cost_floor − cost_π) / (cost_floor − cost_oracle)
```

- Oracle = (s,S) per SKU tuned on the **realised** scene with hindsight, jointly
  under the capacity constraint (coordinate descent).
- On shock scenes the oracle is **piecewise**: separate (s,S) before and after the
  known change point. A single static (s,S) is not a ceiling for an adaptive
  policy on a regime shift.
- It is still a heuristic, not a true optimum. **Values > 100% are possible and are
  reported, not clipped.**
- If `cost_floor − cost_oracle` is < 2% of floor cost on a scene family, % gap is
  unstable there: report raw relative cost instead and flag it.

---

## 6. PPO

- Same obs, flattened, normalised with fixed config-derived scales.
- Action: `MultiDiscrete` per SKU, order quantity in 16 bins spanning
  `[0, 2 · μ_i · (L̄_i + 1)]`.
- `sb3` PPO, vectorised envs sampling train-family seeds, ≥5 training seeds.
- Two variants: **PPO-stable** (train family only, same data as everyone else) and
  **PPO-randomised** (domain randomisation over demand level / lead-time mean, still
  excluding the held-out shock *shapes*). Without the second, "PPO degrades on
  shocks" is true by construction.
- Reward = negative daily cost / fixed scale (sums to −episode cost / scale).
- Gets the same tuning budget discipline as everything else: validation scenes only.

---

## 7. LLM pipeline

### 7.1 Prompt

The prompt describes the **environment spec** (costs, capacity, number of SKUs,
nominal lead times, obs schema, function signature) — drawn from a distribution of
train-family configs so prompts vary. It never mentions held-out shock types. The
LLM's pretrained knowledge that disruptions exist is part of the hypothesis, and the
writeup says so.

### 7.2 Program search (FunSearch-style)

- Islands of programs; each step prompts the LLM with 2 parents + scores, asks for
  an improved program; keep if it beats the island's worst.
- Fitness = verifier score on a **fixed pool of train scenes** (CRN).
- Dedupe by normalised AST hash.
- Budget counted in **verifier evaluations**, so methods are compared at matched
  budget.
- Run with the frontier API model; also run with the small base model, to separate
  "the search" from "the teacher model".

### 7.3 SFT + GRPO

- **SFT**: LoRA on Qwen-coder 1.5B → 7B, on deduplicated top programs from search,
  paired with varied environment-spec prompts. Goal: valid, reasonable code from
  the start. Track diversity so GRPO does not start collapsed.
- **GRPO**: TRL `GRPOTrainer` + vLLM generation, LoRA, group size 8.
  Reward per completion = mean over **S fresh train scenes** (default 8) of
  `clip((cost_base − cost_π) / cost_base, −1, 1)`; invalid or timed-out program = −1.
  Baseline = tuned static base-stock, costs cached per scene in a fixed pool.
- Needs a Linux GPU box. Not feasible on the Mac.

### 7.4 Decoding protocol

- **Greedy**: one program at temperature 0.
- **Sampled best-of-k**: k programs, the best on **validation** scenes is selected,
  then evaluated on held-out. Selection never sees held-out.
- Report both, plus validity rate.

---

## 8. Verifier and sandbox

The verifier runs untrusted model-written code and its score is the reward, so it
is a security boundary *and* a reward-hacking boundary.

- **Sim stays in the trusted parent.** The sandboxed child receives only obs dicts
  and returns orders. It never holds the scene, so it cannot read future demand.
- **Static checks** (AST): imports allowlisted (`math`, `statistics`); no dunder
  attribute access, no `eval/exec/open/compile/globals/getattr/__import__`.
- **Runtime limits**: per-call and per-episode time budget, memory rlimit, CPU
  rlimit, no network (container `--network none` on the training box).
- **Batching**: one IPC round-trip per day carries obs for a batch of scenes, so
  sandboxing does not kill throughput.
- **Adversarial test suite**: programs that try imports, introspection, infinite
  loops, huge allocations, invalid outputs, mutation of obs, "never order",
  "flood the warehouse". Each must be blocked or scored correctly.
- Human-written code in the game is local-only unless a real container sandbox
  (e.g. gVisor) is in place.

Throughput target: **≥ 2,000 episodes/min** (N=8, T=365) for trusted policies on
one core. Measured in Phase 0, vectorised with numpy only if missed.

---

## 9. Scene families

| Family | Split | Definition |
|---|---|---|
| `stable` | train / val / in-dist test | Negative-binomial demand around μ_i with weekly seasonality; lead time ~ nominal ± jitter |
| `demand_spike` | held-out | At t0 ∈ [120, 240], demand mean × m (m ∈ [2, 4]) for a random SKU subset; persists |
| `supplier_disruption` | held-out | At t0, lead-time mean silently jumps (e.g. 3 → 10 days) for a SKU subset |
| `new_sku` | held-out (optional) | A SKU enters mid-horizon with empty history |

Capacity is set so that summed base-stock targets exceed `C` by ~20–30% at nominal
load: the coupling must bind, or the problem degenerates to N independent textbook
problems ("sim too easy" risk).

### 9.1 Frozen evaluation scenes

- Val, in-dist test, and held-out test scenes are generated once, saved as `.npz`,
  and listed in `scenes/manifest.json` with SHA-256 per file.
- A pytest test regenerates them from seeds and checks the hashes: generator drift
  fails the test suite.
- Every result file records the manifest hash it was computed on.

---

## 10. Evaluation protocol (fixed before any training)

- **Paired**: every policy on the same scenes. Report per-scene differences vs the
  floor, mean with paired bootstrap 95% CI, win rate.
- **Learned methods** (PPO, search, GRPO): distribution across ≥5 training seeds /
  search runs, not the best run.
- **Families reported separately**: in-dist test vs each held-out family.
- **Metrics**: total cost + 5-way breakdown, fill rate, scrap units, % gap closed,
  validity rate, invalid/timeout episodes.
- **Matched budget** for search vs GRPO: verifier evaluations spent.

---

## 11. Real-trace judge (M5)

- Demand traces from M5 (Walmart daily unit sales). Lead times, costs and capacity
  stay **synthetic**, sampled from the train-family priors. The writeup says so.
- SKU groups of N = 8 from the same store + department, filtered to items with
  enough volume (M5 is highly intermittent).
- **Censoring caveat**: M5 records sales, not demand. On stockout days true demand
  is under-counted. Stated openly; optionally drop items with suspicious zero runs.
- Output: policy ranking on M5 vs ranking in sim, **Kendall τ with bootstrap CI**.
- License/terms of M5 data to be checked before use (Phase 1).

---

## 12. Game — "Run the warehouse"

- Player manages 3–4 SKUs, ordering **weekly** (~26 turns, under 5 minutes).
- Screen: on-hand, pipeline, demand chart, capacity bar, running cost breakdown.
- A shock hits mid-game (spike or disruption), unannounced — like the held-out sets.
- End screen: player vs base-stock vs PPO vs best LLM program on the **same scene**
  (CRN makes it fair), with the winning program's code shown and annotated.
- Seeded daily challenge for shareable scenes.
- Optional local-only "write your own policy" mode, scored by the verifier.
- Streamlit; game reuses the sim with weekly decisions wrapped over daily steps.

---

## 13. Repository layout

```
src/reordergym/
  config.py            # Pydantic: SkuParams, WarehouseConfig, ScenarioSpec
  scenes.py            # generator, families, freeze + manifest
  sim/
    core.py            # daily step, cost accounting
    obs.py             # the one obs builder (all policies, game, prompts)
  policies/
    classical.py       # base-stock, (s,S), adaptive base-stock
    oracle.py          # hindsight (piecewise) oracle
    tuning.py
  verifier/
    sandbox.py         # AST checks, worker process, limits
    score.py           # multi-scene reward
  rl/ppo.py
  llm/
    prompts.py
    search.py          # FunSearch-style
    sft.py, grpo.py
  eval/
    evaluate.py, stats.py, report.py
  real/m5.py
  game/app.py
scenes/                # frozen eval scenes + manifest (or regenerated + hashed)
configs/
tests/
```

Dependency groups keep the core light: core (numpy, pydantic), `rl` (sb3, torch),
`llm` (trl, peft, vllm — Linux GPU only), `game` (streamlit, plotly).

---

## 14. Invariants we test

1. Hand-traced micro scene (2 SKUs, 10 days): exact receipts, scrap split, lost
   sales and every cost component, written by hand first.
2. Inventory conservation per SKU: start + received = sold + scrapped + end on-hand.
3. Capacity: end-of-receipt on-hand never exceeds `C`.
4. Determinism: same scene + policy ⇒ identical trajectory and cost.
5. CRN: an order placed on day t always gets `lead_time[i,t]`, regardless of policy.
6. No leakage: obs contains no `t`, `T`, future arrays, or arrival days (schema test).
7. Newsvendor sanity: 1 SKU, lead time 1 (next-morning delivery), no fixed cost, no
   capacity → tuned base-stock matches exhaustive search over S, and the analytic
   critical fractile (derived in the test docstring).
8. Uncapacitated multi-SKU → SKUs are independent (cost = sum of single-SKU runs).
9. Verifier: adversarial suite is blocked; scores match direct trusted execution.
10. Frozen scenes: regenerated hashes match the manifest.
11. Cost of oracle ≤ cost of static (s,S) floor on every scene it was tuned on.
