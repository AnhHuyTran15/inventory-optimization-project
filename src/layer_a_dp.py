"""
layer_a_dp.py
=============
Single-echelon stochastic (s, S) inventory policy via infinite-horizon
value iteration (Bellman equation). See notebooks/02_layer_a_dp_inventory.ipynb
for the full narrative, including the 3 debugging iterations that led to
this final formulation (parameters matter a lot for whether ordering is
ever optimal - documented there, not repeated here).

Core function: solve_infinite_horizon() -> (s, S) for one store-item pair.
run_all_pairs() sweeps every row of model_inputs.csv and logs progress.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.logging_config import get_logger, log_run_event

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_INPUTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_inputs.csv"
POLICY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "layer_a_policies.csv"

GAMMA = 0.99
TOL = 1e-4
MAX_ITER = 3000
STOCKOUT_MARGIN_RATE = 0.30  # assumption: stockout cost = 30% of unit cost (lost margin/goodwill)


def discretize_normal_demand(mean: float, std: float, n_std: float = 4.0):
    max_d = max(int(mean + n_std * std), 5)
    xs = np.arange(0, max_d + 1)
    cdf_hi = norm.cdf(xs + 0.5, mean, std)
    cdf_lo = norm.cdf(xs - 0.5, mean, std)
    pmf = cdf_hi - cdf_lo
    pmf[0] += norm.cdf(0.5, mean, std) - pmf[0] + norm.cdf(-0.5, mean, std)
    pmf = np.clip(pmf, 0, None)
    return xs, pmf / pmf.sum()


def solve_infinite_horizon(demand_mean, demand_std, K, h, p, x_max,
                            gamma=GAMMA, tol=TOL, max_iter=MAX_ITER):
    """
    Value iteration for the stationary (s, S) policy.
    Returns: Y (state grid), order_mask, a_star (order quantity per state),
             S (order-up-to level), iterations run, final convergence error.
    """
    xs_d, pmf_d = discretize_normal_demand(demand_mean, demand_std)
    Y = np.arange(0, x_max + 1)
    V = np.zeros(x_max + 1)

    end_inv = np.maximum(Y[:, None] - xs_d[None, :], 0).astype(int)
    shortage = np.maximum(xs_d[None, :] - Y[:, None], 0)
    immediate = h * end_inv + p * shortage

    it, diff = 0, np.inf
    for it in range(max_iter):
        future = V[np.minimum(end_inv, x_max)]
        G = np.sum(pmf_d[None, :] * (immediate + gamma * future), axis=1)
        y_star = int(np.argmin(G))
        cost_order = K + G[y_star]
        V_new = np.minimum(G, cost_order)
        diff = float(np.max(np.abs(V_new - V)))
        V = V_new
        if diff < tol:
            break

    order_mask = cost_order < G
    S = y_star
    a_star = np.where(order_mask, S - Y, 0)
    return Y, order_mask, a_star, S, it, diff


def check_sS_structure(Y, a_star, s, S) -> int:
    """Returns count of states that violate the pure (s,S) shape (should be 0)."""
    violations = 0
    for x, a in zip(Y, a_star):
        expected = max((S - x) if x <= s else 0, 0)
        if a != expected:
            violations += 1
    return violations


def solve_for_row(row: pd.Series, x_max: int | None = None) -> dict:
    """Solve Layer A for a single model_inputs.csv row, returns a result dict."""
    demand_mean, demand_std = row["demand_mean"], row["demand_std"]
    unit_cost = row["unit_cost_placeholder"]
    K = row["order_cost_fixed"]
    h = unit_cost * row["holding_cost_rate_annual"] / 365.0
    p = STOCKOUT_MARGIN_RATE * unit_cost

    if x_max is None:
        # heuristic sizing: EOQ order-of-magnitude + buffer, capped for runtime safety
        eoq_estimate = np.sqrt(2 * K * demand_mean / max(h, 1e-6))
        x_max = int(min(max(eoq_estimate * 1.5, 200), 3000))

    Y, order_mask, a_star, S, it, diff = solve_infinite_horizon(
        demand_mean, demand_std, K, h, p, x_max
    )
    s = int(Y[order_mask].max()) if order_mask.any() else -1
    violations = check_sS_structure(Y, a_star, s, S)
    converged = diff < TOL

    return {
        "store": row["store"], "item": row["item"],
        "s": s, "S": S, "x_max_used": x_max,
        "iterations": it, "final_diff": diff, "converged": converged,
        "violations": violations, "n_states": len(Y),
    }


def run_all_pairs(model_inputs_path: Path = MODEL_INPUTS_PATH, save: bool = True) -> pd.DataFrame:
    """Sweep every store-item pair, log progress every 50 pairs, save results."""
    df = pd.read_csv(model_inputs_path)
    logger.info("Layer A: solving DP for %d store-item pairs", len(df))

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = solve_for_row(row)
        results.append(result)
        if result["violations"] > 0:
            logger.warning("Store %s Item %s: %d/%d states violate (s,S) structure",
                           row["store"], row["item"], result["violations"], result["n_states"])
        if (i + 1) % 50 == 0:
            logger.info("Layer A progress: %d/%d pairs solved", i + 1, len(df))
            if save:
                POLICY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(results).to_csv(POLICY_OUTPUT_PATH, index=False)

    results_df = pd.DataFrame(results)

    if save:
        POLICY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(POLICY_OUTPUT_PATH, index=False)
        logger.info("Saved Layer A policies to %s", POLICY_OUTPUT_PATH)

    log_run_event(
        "layer_a_batch_completed",
        n_pairs=len(results_df),
        n_converged=int(results_df["converged"].sum()),
        n_with_violations=int((results_df["violations"] > 0).sum()),
        mean_iterations=float(results_df["iterations"].mean()),
    )
    return results_df


if __name__ == "__main__":
    run_all_pairs()
