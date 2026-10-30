# FabDispatch — End-to-End Design

Status: draft for review. Nothing here is implemented yet.

## 0. Research question

> Can a masked-PPO dispatching policy reduce weighted tardiness and flow time
> versus strong dispatching rules, and under which conditions (load, failures,
> due-date tightness, urgent orders) does it help or hurt?

Success = a trustworthy, paired comparison with uncertainty estimates — not "PPO wins".

Scope disclaimer: a simplified, semiconductor-*inspired* job shop. Not a fab digital
twin, not a real deployment.

---

## 1. System at a glance

```
ScenarioSpec (Pydantic) + seed
        │
        ▼
ScenarioGenerator ──► ScenarioInstance  (ALL exogenous randomness, pre-sampled)
        │                  jobs, realized durations, failure schedules
        ▼
Simulator (SimPy)  ◄── one core, shared by every policy
   advance() -> Decision | None      apply(decision, job_id)
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
RuleRunner (FIFO/SPT/EDD/ATCS)   FabDispatchEnv (Gymnasium)
   │                              │  features + mask + reward
   │                              ▼
   │                         MaskablePPO training ──► MLflow (runs, checkpoints)
   ▼                              │
Evaluator  ◄──────────────────────┘   paired held-out scenarios, stress suites
   │
   ▼
Report / Dashboard (Streamlit)        FastAPI inference service (frozen model,
                                       rule fallback) ◄── simulator as client
```

The single most important structural decision: **one simulator core with an
`advance()/apply()` interface**. Rules, the Gym env, the evaluator, and the API
client all drive the same loop, so the only thing that differs between policies is
the choice of job. Nothing else may differ.

---

## 2. Domain model

Time unit: minutes (float). All entities are Pydantic models (config) or plain
dataclasses (hot-path runtime state) — Pydantic validation at boundaries, not in the
inner loop.

| Entity | Fields (MVP) |
|---|---|
| `ProductFamily` | id, route: list of `Operation` |
| `Operation` | stage id, mean processing time per eligible machine |
| `Machine` | id, stage id, speed factor, failure params, setup time on family change |
| `Job` | id, family, release time `A_j`, due date `d_j`, weight `w_j`, op index |
| `Decision` | time, machine id, candidate job ids (canonical order), mask |

Routing: each family has 3–5 operations over 3 stages; one family has a re-entrant
visit (e.g. stage 1 → 2 → 1 → 3) — the semiconductor-flavoured part.

Due dates: `d_j = A_j + k · TWK_j`, where `TWK_j` is total expected work, `k` is the
tightness knob (lower = tighter). Weights: e.g. {1: 70%, 2: 20%, 4: 10%}; urgent
orders get high weight and small `k`.

---

## 3. Simulator

### 3.1 Why SimPy, and how we use it

SimPy supplies the clock, the event queue, timeouts, and `Process.interrupt()`
(which models a breakdown well). We do **not** use `simpy.Resource` for machines: its
queue discipline is fixed, and here the queue discipline is the policy. Machines are
our own state objects; SimPy only schedules events.

Pausing for RL: dispatch requests are pushed onto a `pending` list. `advance()`
calls `env.step()` until `pending` is non-empty or the event queue is empty. That
gives Gym-style control without threads or callbacks.

### 3.2 Event types

job release · operation complete · machine failure · machine repair · (later) urgent
order injection.

### 3.3 Decision points

A decision exists when a machine is **idle, up, and its stage queue is non-empty**.
Triggers: release into a queue, operation complete (frees the machine and may feed
a downstream queue), repair.

- **Simultaneous decisions**: resolved in ascending machine id. After each
  assignment the remaining candidates are recomputed (a job taken by M1 disappears
  from M2's set).
- **Work-conserving**: if work exists, something is dispatched. Limitation, stated
  openly: the policy cannot learn strategic idling (e.g. waiting for an urgent job
  to arrive).
- **Trivial decisions** (exactly 1 candidate) are auto-applied for every policy and
  never reach the agent. Cost still accrues through elapsed time. This keeps PPO
  steps meaningful and changes no policy's outcome.

### 3.4 Failures

- Per-machine failures on **calendar time** (exponential TTF, lognormal TTR),
  pre-sampled per machine.
- A failure during processing **preempts and resumes**: remaining work continues on
  the same machine after repair. No scrap and no rework in the MVP.
- A failure while idle just makes the machine unavailable.

Calendar-time failures are a choice made for paired evaluation: failure timing then
does not depend on policy. Busy-time failures are more realistic but couple
disturbances to the policy. That can be a later ablation.

### 3.5 Setups

Sequence-dependent: a setup of `s_m` minutes is incurred when machine `m` switches
product family. Initial setup state = "none" (the first job always pays it, or we
seed initial states — decide in config).

### 3.6 Event log

Every state change is one typed `EventRecord(time, kind, machine, job, op, detail)`.
The log is the ground truth for metrics, tests, the Gantt chart, and debugging.
Metrics are computed **from the log**, not from counters spread through the sim.

---

## 4. Randomness and common random numbers (paired evaluation)

All exogenous randomness is sampled **before** the simulation starts into a
`ScenarioInstance`:

| Source | Sampled as | Keyed by |
|---|---|---|
| Arrivals, families, weights, due dates | full job list | scenario seed, stream "jobs" |
| Processing time noise | multiplier `ε_{j,o}` per (job, op) | stream "proc" |
| Failures | (fail time, repair duration) list per machine | stream "fail:{m}" |

Realized duration = `mean(op, machine) / speed(machine) · ε_{j,o}`. Because noise
is per (job, op) and not drawn "on the next call to rng", **dispatch order cannot
change what randomness a job gets**. Every policy faces exactly the same jobs and
disturbances. Streams come from `numpy.random.SeedSequence(seed).spawn(...)`.

Consequence: the simulator itself is deterministic given an instance. Same instance
+ same policy ⇒ byte-identical event log (a test).

---

## 5. Candidates, observation, action

### 5.1 Candidate set (shared by ALL policies)

Candidates for machine `m` = jobs waiting at `m`'s stage whose current op is
eligible on `m`, sorted canonically by `(queue_entry_time, job_id)` and truncated to
`K` slots.

- Rules choose from the **same** truncated set as PPO, so truncation cannot
  quietly be the real scheduler.
- `K` is chosen so truncation is rare at nominal load (start with K=16). We log the
  **truncation rate** as a metric; if it is non-trivial in a scenario, the results
  for that scenario come with a caveat.

### 5.2 Observation (no future leakage)

- **Machine block**: stage one-hot, current setup family one-hot, speed.
- **Candidate table `K × F`** per slot: family one-hot, *expected* proc time on m,
  setup-needed flag, weight, slack = `d_j − t − expected remaining work`, time in
  queue, remaining ops, is-re-entrant flag, downstream queue length for its next
  stage.
- **Global block**: WIP, queue length per stage, machines up/down per stage, fraction
  of release window elapsed.
- **Mask**: `K` booleans.

Never observed: realized `ε`, future failures, future arrivals.

Features are normalised with fixed scales derived from the config (not running
statistics). That makes serving deterministic, and the scales are versioned with
the model.

`features.py` is the only place features are built. The Gym env and the API both
import it. This is the training/serving parity guarantee.

Known limitation: a flat MLP over ordered slots can learn position bias. Later
improvement: a permutation-invariant scorer (a shared MLP per candidate → logits).
Don't do it in the MVP; measure first.

### 5.3 Action

`Discrete(K)` = slot index, with invalid slots masked (`MaskablePPO`,
`action_masks()`). Invalid-action rate is still tracked: it must be 0, and the
metric proves it.

---

## 6. Objective and reward

### 6.1 Objective (minimise)

```
J = α Σ_j (C_j − A_j)  +  β Σ_j w_j · max(0, C_j − d_j)  +  λ Σ_k s_k
```

(The setup weight is λ, not γ, to avoid a clash with the RL discount γ.)

### 6.2 Exact incremental reward

Define the partial cost at time t over released jobs:

```
J(t) = Σ_j [ α·(min(C_j,t) − A_j)  +  β·w_j·max(0, min(C_j,t) − d_j) ]  +  λ·(setup time started by t)
```

Step reward between decisions at `t_k` and `t_{k+1}`:
`r_k = −(J(t_{k+1}) − J(t_k)) / scale`.

Properties:
- Telescopes: `Σ_k r_k = −J / scale` exactly when the episode drains. **This is
  a unit test**, and it checks the reward against an independent metrics
  computation from the event log.
- Charges elapsed simulated time, not number of decisions. A policy is not
  penalised for "more decision events".
- Due dates crossed mid-interval are handled exactly by the `max`.
- `scale` = a config constant (e.g. n_jobs × mean op time), versioned.

Note: setup time already inflates flow time, so λ is an *extra* penalty on
changeovers (e.g. tool wear / qualification cost). Start with λ small or 0 and treat
it as a knob.

### 6.3 Episode boundaries

- Arrivals stop at `T_release`. The sim then drains all jobs → `terminated=True`.
- Safety cap `T_max`: `truncated=True`. Cost up to `T_max` has already been charged,
  and unfinished jobs are reported as a metric. In evaluation, any truncated episode
  is a failure flag, not a silently shorter run.

### 6.4 Discounting (semi-MDP caveat)

Decisions are unevenly spaced in time, so a per-step γ is a per-*decision* discount.
Use γ ≈ 0.999 (near-undiscounted episodic) with GAE λ ~0.95. Time-aware
discounting `γ^{Δt}` is a possible later ablation, not MVP.

---

## 7. Baselines

All implement `Policy.choose(decision, obs_context) -> slot`. Ties are broken by
canonical order.

| Rule | Picks |
|---|---|
| FIFO | earliest arrival to the current queue |
| SPT | shortest expected proc time on this machine |
| EDD | earliest due date |
| ATCS | Apparent Tardiness Cost with Setups (weights, slack, setup); k₁, k₂ tuned on *train* seeds only |
| Random | uniform over valid slots (sanity floor) |

ATCS is the strong rule PPO has to beat. Tuning it only on train seeds keeps the
comparison fair.

Later: CP-SAT (OR-Tools) on small instances as a near-optimal reference, plus a
rolling-horizon variant.

---

## 8. Evaluation protocol (fixed before any training)

- **Seed splits** (disjoint): train scenario seeds `[0, 10k)`, validation
  `[10k, 10.1k)`, test `[20k, 20.2k)`.
- **Training seeds**: ≥5 independent PPO runs. Report the distribution across runs,
  not only the best run.
- **Paired design**: every policy runs on the same test instances. Report the per-instance
  difference vs ATCS, the mean with a bootstrap 95% CI, and the win rate.
- **Suites**: nominal · high load · frequent failures · tight due dates · shifted
  product mix · urgent-order injection. PPO is trained on nominal (or a mixture,
  decided up front), and the other suites are reported as OOD.
- **Metrics** (from the event log): weighted tardiness, on-time rate, mean and p95 flow
  time, throughput within `[0, T_release]`, total setup time, makespan, decision
  latency, invalid-action rate, truncation rate, timeout episodes.
- **Ablations (Phase 5)**: remove feature groups (slack, setup flag, downstream
  congestion), and compare K values.

---

## 9. Training & MLOps

- `sb3_contrib.MaskablePPO`, `MlpPolicy`, vectorized envs (`SubprocVecEnv`),
  each env sampling train seeds.
- MLflow run records: config hash, git SHA, scenario spec, obs-schema version, feature
  scales, reward scale, hyperparameters, learning curves, periodic validation KPIs, and
  the final checkpoint.
- Model package = `{weights, obs_schema_version, feature_config, K, policy metadata}`.
  The API refuses a package whose schema version does not match.

---

## 10. Serving

- **FastAPI** `POST /v1/dispatch`: the request is a raw `DecisionContext` (machine state +
  candidate jobs + global state as Pydantic models, including `schema_version`). The server
  builds features with the shared `features.py`, runs the frozen policy, and returns
  `{job_id, slot, model_version, fallback_used, latency_ms}`.
- Validation: malformed input → 422; schema/model mismatch → 409; empty candidates
  → 422.
- **Fallback**: the client (simulated dispatcher) uses ATCS on timeout/5xx. The
  server uses ATCS if inference raises. Fallback use is counted.
- `GET /healthz`, `GET /v1/model`. Prometheus-style counters (latency, fallback
  count, action distribution) come later.
- Docker: one image, separate entrypoints for `api`, `train`, `eval`, `dashboard`.
  Inference never trains.

## 11. Dashboard (Streamlit + Plotly)

Gantt from the event log, WIP/queue-length time series, a KPI table per policy, a
paired-difference plot with CIs, and "replay this test scenario under policy X".

---

## 12. Repository layout

```
src/fabdispatch/
  config.py          # Pydantic: FactoryConfig, WorkloadConfig, ScenarioSpec
  scenario.py        # ScenarioSpec + seed -> ScenarioInstance (pre-sampled)
  sim/
    state.py         # runtime Job/Machine state
    events.py        # EventRecord, EventLog
    simulator.py     # SimPy core: advance()/apply()
  dispatch/
    candidates.py    # canonical candidate set (shared)
    rules.py         # FIFO, SPT, EDD, ATCS, Random
    runner.py        # run(instance, policy) -> EventLog
  metrics.py         # KPIs from EventLog; J(t)
  rl/
    features.py      # shared obs builder (env + API)
    env.py           # FabDispatchEnv (Gymnasium)
    train.py
  eval/
    suites.py, evaluate.py, report.py
  serve/
    api.py, schemas.py, client.py
  dashboard/app.py
tests/
configs/             # YAML scenario specs, versioned
docs/design.md
```

Tooling: `uv`, Python 3.12 (safest for torch/SB3 wheels), ruff, mypy (strict on
`src/`), pytest + hypothesis for invariants, GitHub Actions.

---

## 13. Invariants we test (property tests over many random seeds)

1. A machine never has two overlapping processing intervals.
2. An op never starts before the job's previous op completes.
3. A job only runs on a machine eligible for its current op.
4. No machine processes while down. Preempted work resumes and total processed time
   = realized duration.
5. Conservation: released = completed + in-system, at every event. After drain,
   completed = released.
6. Determinism: same instance + policy ⇒ identical event log.
7. CRN: the realized duration of (job, op) is identical across policies.
8. Reward telescoping: Σ rewards = −J/scale, where J comes from the event log.
9. Hand-traced micro scenario (2 machines, ~10 jobs, deterministic): exact expected
   schedule and tardiness for FIFO and EDD.
10. Env: random valid actions always reach termination, and the mask always has ≥1
    valid slot when a decision is presented.
11. API: rejects malformed / schema-mismatched requests, and fallback triggers on
    timeout.
