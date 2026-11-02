# Lessons

## Inventory timing
- If an order placed tonight arrives before demand on day L, the order-up-to window is L days, not L+1.

## Keep it minimal
- Rule: validate only what would cause a real, silent bug in the next step. Internal data types get 1–3 checks. Add fields and checks when a concrete need appears, not in anticipation. Prefer type hints (e.g. `Literal`) over runtime checks.
