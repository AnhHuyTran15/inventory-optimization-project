"""
layer_b_pooling.py
===================
Multi-echelon risk-pooling analysis (Eppen, 1979 style base-stock
comparison). Answers: how much safety stock is saved by holding
inventory centrally (pooled across correlated stores) vs. each store
holding its own buffer independently?

See notebooks/03_layer_b_risk_pooling.ipynb for the full walkthrough,
including how the correlation matrix is computed from the RAW train.csv
(not from model_inputs.csv, which only has marginal mean/std per pair -
correlation requires the full time series across stores for a given item).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.logging_config import get_logger, log_run_event

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def compute_correlation_matrix(item_id: int, demand_path: Path = DEMAND_PATH) -> pd.DataFrame:
    """
    Pivot to (date x store) for a single item, return the store x store
    Pearson correlation matrix of daily demand.
    """
    df = pd.read_csv(demand_path, parse_dates=["date"])
    item_df = df[df["item"] == item_id]
    pivot = item_df.pivot(index="date", columns="store", values="sales")
    corr = pivot.corr()
    logger.info("Item %d: correlation matrix computed across %d stores", item_id, corr.shape[0])
    return corr


def decentralized_safety_stock(sigmas: np.ndarray, z: float, lead_time_days: float) -> float:
    """Sum of independently-held safety stock, ignoring correlation (worst case, no pooling)."""
    return z * np.sqrt(lead_time_days) * np.sum(sigmas)


def pooled_safety_stock(sigmas: np.ndarray, corr_matrix: np.ndarray,
                         z: float, lead_time_days: float) -> float:
    """
    Safety stock if inventory is held centrally: uses the pooled variance
    sqrt(sum(sigma_i^2) + sum_{i!=j} rho_ij * sigma_i * sigma_j), which is
    <= sum(sigma_i) whenever correlations are < 1 - this inequality IS the
    risk-pooling effect, not an assumption.
    """
    cov_matrix = np.outer(sigmas, sigmas) * corr_matrix
    pooled_variance = np.sum(cov_matrix)
    return z * np.sqrt(lead_time_days) * np.sqrt(max(pooled_variance, 0))


def compare_scenarios(item_id: int, model_inputs: pd.DataFrame,
                       service_level: float = 0.95) -> dict:
    """
    For a given item, compare decentralized vs pooled safety stock across
    all 10 stores, using the item's real demand correlation structure.
    """
    item_rows = model_inputs[model_inputs["item"] == item_id].sort_values("store")
    sigmas = item_rows["demand_std"].to_numpy()
    lead_time = item_rows["lead_time_mean_days"].mean()  # approx: avg across assigned vehicle types
    z = norm.ppf(service_level)

    corr_df = compute_correlation_matrix(item_id)
    # align order with item_rows["store"]
    stores = item_rows["store"].to_numpy()
    corr_matrix = corr_df.loc[stores, stores].to_numpy()
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # missing pairs -> assume uncorrelated

    ss_decentralized = decentralized_safety_stock(sigmas, z, lead_time)
    ss_pooled = pooled_safety_stock(sigmas, corr_matrix, z, lead_time)
    pct_savings = 100 * (ss_decentralized - ss_pooled) / ss_decentralized if ss_decentralized > 0 else 0.0

    result = {
        "item": item_id,
        "n_stores": len(sigmas),
        "ss_decentralized": round(ss_decentralized, 1),
        "ss_pooled": round(ss_pooled, 1),
        "pct_savings": round(pct_savings, 1),
        "mean_pairwise_correlation": round(
            float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])), 3
        ),
    }
    logger.info("Item %d pooling: decentralized=%.1f pooled=%.1f savings=%.1f%%",
               item_id, ss_decentralized, ss_pooled, pct_savings)
    return result


def run_all_items(model_inputs_path: Path, save: bool = True) -> pd.DataFrame:
    """Sweep every item (1-50), compute pooling comparison, save results."""
    model_inputs = pd.read_csv(model_inputs_path)
    items = sorted(model_inputs["item"].unique())
    logger.info("Layer B: running risk-pooling comparison for %d items", len(items))

    results = [compare_scenarios(item_id, model_inputs) for item_id in items]
    results_df = pd.DataFrame(results)

    if save:
        out_path = PROJECT_ROOT / "data" / "processed" / "layer_b_pooling_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info("Saved Layer B results to %s", out_path)

    log_run_event(
        "layer_b_batch_completed",
        n_items=len(results_df),
        mean_pct_savings=float(results_df["pct_savings"].mean()),
    )
    return results_df


if __name__ == "__main__":
    from src.data_prep import OUTPUT_PATH as MODEL_INPUTS_PATH
    run_all_items(MODEL_INPUTS_PATH)
