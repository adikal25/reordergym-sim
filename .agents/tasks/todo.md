# FabDispatch — Task Plan

Design: `docs/design.md`. Each phase has a definition of done (DoD). Don't start a phase
until the previous DoD holds.

## Phase 0 — Foundation micro-sim (first build target)
- [ ] Project scaffold: uv, Python 3.12, ruff, mypy, pytest, CI skeleton
- [ ] Pydantic config models (minimal: machines, families/routes, jobs)
- [ ] Event log types (`EventRecord`, `EventLog`)
- [ ] SimPy core with `advance()/apply()` — 2 machines, deterministic times, no failures
- [ ] Canonical candidate set
- [ ] FIFO and EDD rules + runner
- [ ] Metrics from event log: flow time, tardiness, weighted tardiness, on-time rate
- [ ] Hand-traced 10-job scenario: expected schedule + tardiness written by hand first
- DoD: the hand-traced test passes for FIFO and EDD; the determinism test passes

## Phase 1 — Full simulator
- [ ] 3 stages / 6 machines / 3 families, re-entrant route
- [ ] Scenario generator with SeedSequence streams (jobs, proc noise, failures)
- [ ] Stochastic processing times (per job-op multiplier)
- [ ] Failures (calendar time, preempt-resume) + repairs
- [ ] Sequence-dependent setups
- [ ] Property tests for invariants 1–7 (hypothesis, many seeds)
- DoD: all invariants hold across ≥500 random seeds; one scenario traced manually

## Phase 2 — Baseline benchmark
- [ ] SPT, ATCS, Random
- [ ] ATCS parameter tuning on train seeds only
- [ ] Stress suites as config files
- [ ] Paired evaluator + bootstrap CI + report (markdown/CSV)
- DoD: a reproducible baseline report from one command

## Phase 3 — RL environment
- [ ] `features.py` (shared), fixed normalisation scales
- [ ] `FabDispatchEnv`: obs, mask, exact incremental reward, terminated/truncated
- [ ] Auto-apply trivial decisions
- [ ] Tests: reward telescoping, random-agent termination, mask validity, `check_env`
- DoD: a random masked agent completes 1000 episodes with zero invariant violations

## Phase 4 — Training
- [ ] MaskablePPO + vec envs + MLflow logging
- [ ] Validation KPI callback; checkpoint selection on val seeds only
- [ ] ≥5 training seeds
- DoD: a reproducible checkpoint and a training report

## Phase 5 — Evaluation
- [ ] Test suites, paired diffs vs ATCS, CIs, win rates
- [ ] Ablations (feature groups, K)
- DoD: an honest results write-up, including where PPO loses

## Phase 6 — Serving & demo
- [ ] FastAPI service, schemas, version checks, fallback
- [ ] Simulated dispatcher client drives the sim end to end via the API
- [ ] Docker image, CI
- [ ] Streamlit dashboard
- DoD: one `docker compose up` runs the API and dashboard, and a policy drives the sim end to end

## Open decisions (need input)
- [ ] Train PPO on nominal only, or a mixture of conditions?
- [ ] Objective weights α, β, λ (proposal: α=1, β=5, λ=0 initially)
- [ ] Initial machine setup state: "none" vs seeded
