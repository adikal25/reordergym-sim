# Lessons

## Explaining concepts to Adithya
- Pattern: dense multi-panel explainer pages with heavy jargon were rejected as "really hard to understand".
- Rule: one concept at a time, plain words, one everyday analogy, the inventory example, tiny visuals. Build each idea on the previous one.
- Rule: answer the concept actually asked (RL concepts ≠ inventory policies).
- Interactive visuals only when asked; keep them to one idea with step-by-step buttons and a caption per step.

## Inventory timing
- If an order placed tonight arrives before demand on day L, the order-up-to window is L days, not L+1. Check formulas against the sim before explaining them.

## Who writes the code
- Pattern: Adithya wants to write the project code themselves; Claude writing whole modules was rejected ("help me code it, I don't want you to do it").
- Rule: act as a mentor. Give one small exercise at a time: goal, file, what to think about, hints, and tests to write. Review what they write. Don't write the implementation unless asked; tiny illustrative snippets are fine.

## Keep it minimal
- Pattern: a Scene `__post_init__` with ~15 checks, array locking and a family whitelist was called over-engineering.
- Rule: validate only what would cause a real, silent bug in the next step. Internal data types get 1–3 checks. Add fields and checks when a concrete need appears, not in anticipation. Prefer type hints (e.g. `Literal`) over runtime checks.
