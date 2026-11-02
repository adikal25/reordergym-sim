# ReorderGym — Task Plan

Design: `docs/design.md`. Lessons: `.agents/tasks/lessons.md`.

Strategy: build the cheapest experiment that can **kill** the hypothesis first
(Milestone 1, ~2 weeks). Only build PPO, GRPO, M5 and the game if it survives.

Conventions for every PR:
- Own branch; local checks green before merge: `ruff check`, `ruff format --check`,
  `mypy` (strict), `pytest`. There is no GitHub CI.
- Tests first for anything with a hand-computable answer.
- New deps go in the dependency group of the phase that needs them.
- Classical baselines and LLM programs share ONE interface: `policy(obs) -> list[int]`.

---

## Milestone 1 — Kill test (sim + baselines + zero-shot/search LLM programs)

### PR 1 — Rename + config models
- [x] Rename package `fabdispatch` → `reordergym` locally (src dir, pyproject, README, tests)
- [x] `config.py` (frozen Pydantic):
  - `SkuParams`: unit_cost, holding, lost_sales_penalty, fixed_order_cost, scrap_cost, q_max, nominal_lead_time (≥ 1)
  - `WarehouseConfig`: skus, capacity (> 0)
  - `SimConfig`: horizon T, warmup W < T, demand_history H, lead_time_history M
- [x] Tests: validators reject L = 0, W ≥ T, empty SKU list
- DoD: all local checks green under the new name

### PR 2 — Scene generator (CRN)
- [ ] `scenes.py`: `Scene` (frozen dataclass): `demand[N,T]`, `lead_time[N,T]` (int arrays, keyed by **placement day**), family, seed, `change_point | None`, affected SKUs
- [ ] `SeedSequence(seed).spawn` → one demand stream + one lead-time stream per SKU
- [ ] Families: `stable` (neg-binomial + weekly seasonality, jittered lead time ≥ 1), `demand_spike`, `supplier_disruption`
- [ ] Tests: deterministic per seed; changing SKU 3's params leaves SKU 0's draws identical; shocks change nothing before `change_point`; lead times ≥ 1
- DoD: scenes for all three families generate and pass tests

### PR 3 — Sim core + obs (the heart; most care here)
- [ ] Write the hand-traced 2-SKU / 10-day scene **first**, expected values by hand: receipts, scrap split, lost sales, all 5 cost components
- [ ] `sim/core.py`: `State`, `Order(sku, qty, placed_day, arrival_day)`, daily step in fixed order (receive → capacity/scrap → demand → holding → order), `CostBreakdown`, `run_episode(scene, cfg, policy) -> EpisodeResult`
- [ ] Warm-up exclusion, terminal valuation at unit cost, strict action validation → invalid episode (never clipped)
- [ ] `sim/obs.py`: the only obs builder; plain dicts/lists; no `t`, `T`, arrival days, or future arrays
- [ ] Property tests (hypothesis, ≥500 seeds): conservation, capacity, determinism, CRN, obs-leakage schema, uncapacitated ⇒ SKUs independent
- [ ] `scripts/bench_sim.py`
- DoD: hand-traced test + all invariants pass
- **Gate**: ≥ 2,000 episodes/min, one core, N=8, T=365, trivial policy. If missed, profile obs building first, then vectorise.

### PR 4 — Classical baselines + tuning
- [ ] `policies/classical.py`, each as `policy(obs)` with a Pydantic params model:
  static base-stock (capacity-aware scaling), static (s,S), capped base-stock, adaptive base-stock (rolling μ̂, σ̂, L̂ from obs windows)
- [ ] `policies/tuning.py`: coordinate/grid search on a fixed CRN pool of train scenes, select on val
- [ ] Newsvendor test: 1 SKU, L = 1, no K, no capacity → tuned S = exhaustive-search S = analytic fractile
- [ ] Naive heuristics (e.g. "order last week's demand") for the weak-baseline gate
- DoD: tuned params saved to `configs/baselines/*.json` with the scene-pool hash

### PR 5 — Frozen scenes, oracle, paired evaluator
- [ ] Split seed ranges: train / val / in-dist test / held-out test
- [ ] `scenes/manifest.json`: SHA-256 of each frozen scene's arrays (manifest committed, arrays regenerated); a pytest test checks drift
- [ ] `policies/oracle.py`: hindsight (s,S) per scene, joint under capacity, piecewise around `change_point`; cached
- [ ] `eval/evaluate.py` → per-scene CSV (cost + breakdown + validity); `eval/stats.py`: paired bootstrap CI, win rate, % gap closed (unclipped, flagged when gap < 2%)
- [ ] `eval/report.py` + CLI `uv run reordergym eval ...` → markdown table naming the manifest hash
- DoD: one command → baseline table for every family
- **Gate (weak baseline)**: no naive heuristic beats the tuned floor on `stable`
- **Gate (sim too easy)**: floor–oracle gap ≥ 5% of floor cost on held-out families; else harden capacity/shocks and re-run

### PR 6 — Verifier v1 (built properly once)
- [ ] `verifier/checks.py`: AST allowlist (`math`, `statistics`), banned names, dunder access, size limit
- [ ] `verifier/sandbox.py`: child process holds only the program; sim stays in parent; one IPC round-trip per day for a batch of scenes; per-call + per-episode timeouts; memory/CPU rlimits
- [ ] `verifier/score.py`: multi-scene score vs cached floor costs; invalid/timeout ⇒ −1
- [ ] Adversarial suite: imports, introspection, infinite loop, big alloc, bad outputs, obs mutation, "never order", "flood warehouse"
- [ ] Control test: classical baselines scored through the sandbox == scored directly
- DoD: adversarial suite blocked; sandbox throughput measured

### PR 7 — LLM programs: zero-shot + small search
- [ ] Add `llm` dependency group (API SDK only; no torch yet)
- [ ] `llm/prompts.py`: env spec + obs schema + signature; varied train configs; **no shock hints**
- [ ] Zero-shot: k samples per model, greedy + best-of-k selected on val; archive every program + tokens + cost
- [ ] `llm/search.py`: FunSearch-style, single population first, budget counted in verifier evals and $; AST-hash dedup; fitness on train pool only
- [ ] Read the top programs by hand; check cost breakdowns for hacks
- DoD: LLM rows in the results table, all families, with CIs

### Milestone 1 decision (write up before continuing)
- **Go** if best LLM program (selected on val) beats the floor (incl. adaptive + capped base-stock) on ≥ 1 held-out family with the CI excluding 0.
- **No-go** otherwise: write up the negative result; reframe or stop. Do not start GPU work.

---

## Milestone 2 — Only if Milestone 1 says go
- [ ] **PPO**: Gym env over the same core + obs; PPO-stable and PPO-randomised; ≥5 seeds; val checkpoint selection
- [ ] **GRPO question, pick one**: (a) does training on stable scenes erode the LLM's built-in shock robustness? or (b) can a small model write good policies for unseen warehouse specs?
- [ ] SFT (LoRA, Qwen-coder 1.5B) on search programs → GRPO with verifier reward; Linux GPU box, container `--network none`
- [ ] Final table, annotated winning programs, greedy vs sampled gap
- [ ] M5 real-trace judge: loader, SKU groups, censoring caveat, Kendall τ (check license first)
- [ ] "Run the warehouse" game (design §12): weekly turns, unannounced shock, end screen vs base-stock / PPO / best program with code shown, daily challenge

## Open decisions (need input)
- [ ] Rename the **GitHub repo** to `reordergym`? (local package rename happens in PR 1 regardless)
- [ ] Frontier model + API budget for PR 7 (default proposal: Claude Opus 5.5, cap ~$50 for Milestone 1)
- [ ] Include the `new_sku` held-out family? (default: no, until Milestone 2)
- [ ] Defaults OK? N=8 SKUs, T=365, warm-up 60, H=56, M=10
