"""
test_layer_a.py
================
Regression test: the DP solve must produce a pure (s,S) policy (0
structural violations) for known-good parameter sets. If this ever fails
after a code change, it likely means someone reintroduced the c*a cost
term or reverted to finite-horizon backward induction without enough
periods - both broke the (s,S) structure during development (see
notebooks/02_layer_a_dp_inventory.ipynb for that history).

Run with: pytest tests/test_layer_a.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.layer_a_dp import solve_infinite_horizon, check_sS_structure  # noqa: E402


def test_small_demand_case_has_zero_violations():
    """Store 6 - Item 5 equivalent: low demand, should converge cleanly."""
    Y, order_mask, a_star, S, it, diff = solve_infinite_horizon(
        demand_mean=13.9, demand_std=5.2, K=50.0, h=0.00548, p=3.0, x_max=700
    )
    s = int(Y[order_mask].max()) if order_mask.any() else -1
    violations = check_sS_structure(Y, a_star, s, S)

    assert violations == 0, f"Expected 0 violations, got {violations}"
    assert diff < 1e-3, "Value iteration did not converge within tolerance"
    assert s < S, "Reorder point must be below order-up-to level"


def test_medium_demand_case_has_zero_violations():
    """Store 1 - Item 2 equivalent: medium demand, larger state space."""
    Y, order_mask, a_star, S, it, diff = solve_infinite_horizon(
        demand_mean=53.1, demand_std=15.0, K=50.0, h=0.00548, p=3.0, x_max=1300
    )
    s = int(Y[order_mask].max()) if order_mask.any() else -1
    violations = check_sS_structure(Y, a_star, s, S)

    assert violations == 0, f"Expected 0 violations, got {violations}"
    assert s < S


def test_degenerate_case_never_orders_when_penalty_too_small():
    """
    Documents the failure mode found during development: if the shortage
    penalty is tiny relative to the fixed order cost, the optimal policy
    is legitimately to never order. This is not a bug - asserting it here
    so the behavior is explicit and intentional, not silently "fixed" later.
    """
    Y, order_mask, a_star, S, it, diff = solve_infinite_horizon(
        demand_mean=13.9, demand_std=5.2, K=50.0, h=0.00548, p=0.1, x_max=700
    )
    assert not order_mask.any(), (
        "Expected no ordering states when shortage penalty << fixed order cost"
    )
