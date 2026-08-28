"""
data_prep.py
============
Builds model_inputs.csv from the two raw sources:
  - data/raw/train.csv                                  (demand)
  - data/raw/Transportation__Logistics_Tracking_Dataset.xlsx  (lead time)

This is the tested, working version from the exploratory phase - see
notebooks/01_data_preparation.ipynb for the narrative walkthrough
(including the noise-filtering decisions and why lead time is *sampled*
per store-item pair rather than measured directly - no shared key exists
between the two datasets).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import get_logger, log_run_event

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGISTICS_PATH = PROJECT_ROOT / "data" / "raw" / "Transportation__Logistics_Tracking_Dataset.xlsx"
DEMAND_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_inputs.csv"

RNG_SEED = 42
MIN_OBS_PER_VEHICLE = 5

COST_ASSUMPTIONS = {
    "holding_cost_rate_annual": 0.20,
    "order_cost_fixed": 50.0,
    "target_service_level": 0.95,
    "unit_cost_placeholder": 10.0,
}


def build_lead_time_stats(path: Path = LOGISTICS_PATH) -> pd.DataFrame:
    """Lead time (days) = Trip End Date - Booking Date, filtered for noise, grouped by Vehicle Type."""
    df = pd.read_excel(path, sheet_name="Primary Data")
    df["Trip End Date"] = pd.to_datetime(df["Trip End Date"], errors="coerce")
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors="coerce")
    df["lead_time_days"] = (df["Trip End Date"] - df["Booking Date"]).dt.total_seconds() / 86400.0

    before = len(df)
    df = df[df["lead_time_days"] > (1 / 24)]
    cap = df["lead_time_days"].quantile(0.99)
    df = df[df["lead_time_days"] <= cap]
    after = len(df)
    logger.info("Lead time: kept %d/%d rows after noise filtering (upper cap = %.1f days)",
                after, before, cap)

    overall = {
        "group": "OVERALL",
        "lead_time_mean_days": df["lead_time_days"].mean(),
        "lead_time_std_days": df["lead_time_days"].std(),
        "n_obs": len(df),
    }
    by_vehicle = (
        df.groupby("Vehicle Type")["lead_time_days"]
        .agg(lead_time_mean_days="mean", lead_time_std_days="std", n_obs="count")
        .reset_index()
        .rename(columns={"Vehicle Type": "group"})
    )
    return pd.concat([pd.DataFrame([overall]), by_vehicle], ignore_index=True)


def sample_lead_time_per_pair(lead_time_stats: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    """
    Assign each store-item pair a lead time distribution by weighted random
    draw from real Vehicle-Type distributions (weights = observation
    frequency). Documented assumption: this is a SIMULATION, not a real
    1:1 join - no shared key exists between the two source datasets.
    """
    valid = lead_time_stats[
        (lead_time_stats["group"] != "OVERALL")
        & (lead_time_stats["n_obs"] >= MIN_OBS_PER_VEHICLE)
        & (lead_time_stats["lead_time_std_days"].notna())
    ].reset_index(drop=True)

    logger.info("Lead time: %d Vehicle Types eligible (>= %d obs) as sampling source",
                len(valid), MIN_OBS_PER_VEHICLE)

    rng = np.random.default_rng(RNG_SEED)
    weights = valid["n_obs"] / valid["n_obs"].sum()
    chosen_idx = rng.choice(valid.index, size=n_pairs, p=weights.values)

    assigned = valid.loc[chosen_idx].reset_index(drop=True)
    return assigned.rename(columns={"group": "assigned_vehicle_type"})[
        ["assigned_vehicle_type", "lead_time_mean_days", "lead_time_std_days"]
    ]


def build_demand_stats(path: Path = DEMAND_PATH) -> pd.DataFrame:
    """Per (store, item): demand mean/std + a quick Poisson-vs-Normal suggestion."""
    df = pd.read_csv(path, parse_dates=["date"])
    grouped = df.groupby(["store", "item"])["sales"]
    stats = grouped.agg(
        demand_mean="mean", demand_std="std",
        demand_min="min", demand_max="max", n_days="count",
    ).reset_index()
    stats["dispersion_index"] = (stats["demand_std"] ** 2) / stats["demand_mean"]
    stats["suggested_distribution"] = np.where(
        stats["dispersion_index"] <= 1.5, "Poisson", "Normal_approx"
    )
    return stats


def build_model_inputs(save: bool = True) -> pd.DataFrame:
    """Full pipeline: lead time stats -> demand stats -> sampled join -> cost assumptions -> save."""
    logger.info("Starting model_inputs build")

    lead_time_stats = build_lead_time_stats()
    demand_stats = build_demand_stats()
    sampled_lt = sample_lead_time_per_pair(lead_time_stats, n_pairs=len(demand_stats))

    model_inputs = pd.concat(
        [demand_stats.reset_index(drop=True), sampled_lt.reset_index(drop=True)], axis=1
    )
    for k, v in COST_ASSUMPTIONS.items():
        model_inputs[k] = v

    model_inputs = model_inputs[[
        "store", "item",
        "demand_mean", "demand_std", "demand_min", "demand_max", "n_days",
        "dispersion_index", "suggested_distribution",
        "assigned_vehicle_type", "lead_time_mean_days", "lead_time_std_days",
        "holding_cost_rate_annual", "order_cost_fixed",
        "target_service_level", "unit_cost_placeholder",
    ]]

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_inputs.to_csv(OUTPUT_PATH, index=False)
        logger.info("Saved model_inputs.csv (%d rows) to %s", len(model_inputs), OUTPUT_PATH)

    log_run_event(
        "data_prep_completed",
        n_pairs=len(model_inputs),
        lead_time_mean_overall=float(lead_time_stats.loc[lead_time_stats["group"] == "OVERALL",
                                                           "lead_time_mean_days"].iloc[0]),
    )
    return model_inputs


if __name__ == "__main__":
    build_model_inputs()
