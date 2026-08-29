"""
app.py
======
Two operational screens plus delivery routing:

  Tab 1 "Replenishment Strategy"  - Layer A (s,S) tank: on-hand + in-transit
  Tab 2 "Risk Pooling"            - Layer B DC vs store cards with demand sparklines
  Tab 3 "Dispatch Routes"         - Layer C OR-Tools CVRP on real logistics GPS

On-hand / in-transit are a deterministic ERP-style snapshot (not a user slider).
The source data has sales, not inventory balances, so the snapshot is seeded per
(store, SKU) and only used to drive the gauge and the double-order guard.
"""

import re
import sys
import textwrap
from pathlib import Path

from collections import Counter
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.demand_forecast import forecast_demand  # noqa: E402
from src.layer_a_dp import solve_for_row  # noqa: E402
from src.layer_b_pooling import (  # noqa: E402
    compute_correlation_matrix,
    decentralized_safety_stock,
    pooled_safety_stock,
)
from src.layer_c_routing import (  # noqa: E402
    AVG_TRUCK_SPEED_KMH,
    DEPOT_DEPARTURE_HOUR,
    KG_PER_UNIT,
    LOGISTICS_PATH,
    STOP_SERVICE_MINUTES,
    build_delivery_requests,
    build_vrp_model,
    get_store_coordinates,
    parse_vehicle_capacity_tonnes,
)
from src.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BRANCH_NAMES = {i: f"Store {i}" for i in range(1, 11)}

WAREHOUSE_ICON = """
<svg viewBox="0 0 64 48" width="44" height="33" aria-hidden="true">
  <polygon points="2,20 32,4 62,20" fill="#1e3a5f"/>
  <rect x="6" y="20" width="52" height="24" fill="#0c447c"/>
  <rect x="12" y="28" width="10" height="16" fill="#93c5fd"/>
  <rect x="27" y="28" width="10" height="16" fill="#93c5fd"/>
  <rect x="42" y="26" width="12" height="18" fill="#f8fafc"/>
  <rect x="45" y="32" width="6" height="12" fill="#cbd5e1"/>
</svg>
"""

STORE_ICON = """
<svg viewBox="0 0 48 48" width="32" height="32" aria-hidden="true">
  <polygon points="4,20 24,6 44,20" fill="#0f766e"/>
  <rect x="8" y="20" width="32" height="22" fill="#115e59"/>
  <rect x="12" y="24" width="8" height="8" fill="#99f6e4"/>
  <rect x="28" y="24" width="8" height="8" fill="#99f6e4"/>
  <rect x="20" y="30" width="8" height="12" fill="#ecfdf5"/>
  <rect x="6" y="18" width="36" height="4" fill="#134e4a"/>
</svg>
"""

st.set_page_config(page_title="Inventory Control", layout="wide")


def render_html(html: str) -> None:
    """Render HTML. Indented f-strings become Markdown code blocks under st.markdown."""
    st.html(textwrap.dedent(html).strip())


st.markdown(
    """
    <style>
      :root {
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --paper: #ffffff;
        --canvas: #f8fafc;
        --accent: #1e3a5f;
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-5: 24px;
        --radius: 8px;
        --shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      html, body, [class*="css"] { color: var(--ink); }
      /* Type scale: page title 35-50px desktop / 28-40px mobile, body 14-20px
         (interaction-heavy dashboard), captions 2px under body. */
      h1 { font-size: 2.5rem !important; font-weight: 650 !important;
           letter-spacing: -0.02em; color: var(--ink) !important; }
      h2, .stMarkdown h2 { font-size: 1.15rem !important; font-weight: 650 !important;
           color: var(--ink) !important; margin-bottom: var(--space-3) !important; }
      h3, h4, .stMarkdown h3, .stMarkdown h4 {
           font-size: 1rem !important; font-weight: 600 !important;
           color: #334155 !important; }
      .stCaption, [data-testid="stCaptionContainer"] { font-size: 14px !important; color: var(--muted) !important; }
      @media (max-width: 640px) {
        h1 { font-size: 1.9rem !important; }
      }
      .stButton > button { border-radius: var(--radius) !important; box-shadow: var(--shadow); }
      .stButton > button[kind="primary"] {
        background: var(--accent) !important; border: none !important; color: #fff !important;
      }
      div[data-testid="stMetric"] {
        background: var(--paper); border-radius: var(--radius); box-shadow: var(--shadow);
        padding: var(--space-3) 10px;
      }
      /* Narrow columns (5-up rows on Tab 3) otherwise truncate the label
         and/or value with an ellipsis. Streamlit nests the actual text in
         stMarkdownContainer > p *inside* the label/value element, and that
         inner p carries its own overflow:hidden/ellipsis/nowrap - overriding
         only the outer element (as a first pass here did) has no effect, so
         every level needs the override. */
      div[data-testid="stMetric"] label,
      div[data-testid="stMetric"] label [data-testid="stMarkdownContainer"],
      div[data-testid="stMetric"] label p {
        font-size: 13px !important; color: var(--muted) !important;
        white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
        word-break: keep-all !important; overflow-wrap: normal !important;
      }
      /* Value stays on one line (word-break would otherwise split numbers
         mid-digit, e.g. "$21,130" -> "$21,13" / "0"). overflow:visible alone
         isn't enough insurance at 5-up column widths - a value that doesn't
         fit spills into the next (opaque) card and gets visually cut off
         there, which reads even worse than an ellipsis. Shrinking the value
         font is what actually keeps it inside its own card at realistic
         widths; overflow:visible just stops an ellipsis appearing if a wider
         window is squeezed narrower than that. */
      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
      div[data-testid="stMetricValue"] p {
        overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important;
        font-size: 1.5rem !important;
      }
      /* delta ("↑ 13787 km") defaults to a very light gray with delta_color="off" - too faint */
      div[data-testid="stMetricDelta"] { color: #334155 !important; }
      div[data-testid="stMetricDelta"] svg { fill: #334155 !important; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important; border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important; background: var(--paper);
      }
      .stTabs [data-baseweb="tab-list"] { gap: var(--space-2); }
      .stTabs [data-baseweb="tab"] { border-radius: var(--radius) var(--radius) 0 0; }

      .pillar-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-4);
                    margin: var(--space-4) 0 var(--space-5); }
      .pillar-card { background: var(--paper); border-radius: var(--radius); box-shadow: var(--shadow);
                     padding: var(--space-4); }
      .pillar-card h3 { margin: 0 0 var(--space-2); font-size: 1rem; font-weight: 650; color: var(--ink); }
      .pillar-card p { margin: 0; font-size: 13px; line-height: 1.45; color: var(--muted); }

      .status-badge { display: inline-block; padding: var(--space-2) 14px; border-radius: var(--radius);
                      font-weight: 650; font-size: 15px; letter-spacing: 0.01em; }
      .status-urgent { background: #f8fafc; color: #334155; box-shadow: var(--shadow); }
      .status-transit { background: #f8fafc; color: #334155; box-shadow: var(--shadow); }
      .status-ok { background: #f8fafc; color: #334155; box-shadow: var(--shadow); }
      .ops-note { background: var(--paper); box-shadow: var(--shadow); border-radius: var(--radius);
                  padding: var(--space-3) var(--space-4); font-size: 14px; color: #334155;
                  margin: var(--space-3) 0; }
      .tank-legend { font-size: 12px; color: var(--muted); margin-top: var(--space-2); }
      .tank-legend span { display: inline-block; width: 12px; height: 12px;
                          border-radius: 2px; margin-right: 6px; vertical-align: middle; }
      .demand-legend { font-size: 12px; color: var(--muted); margin: 2px 0 10px; }
      .demand-legend b { display: inline-block; width: 16px; height: 3px; margin: 0 6px 0 12px;
                         vertical-align: middle; border-radius: 1px; }
      .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center;
                     text-align: center; padding: 48px 16px; color: var(--muted); }
      .empty-state svg { margin-bottom: var(--space-3); }
      .empty-state .empty-title { font-size: 1rem; font-weight: 650; color: var(--ink); margin-bottom: var(--space-2); }
      .empty-state .empty-body { font-size: 13px; max-width: 420px; line-height: 1.45; }
      .inline-empty { display: flex; align-items: center; gap: var(--space-2); color: var(--muted);
                      font-size: 13px; padding: var(--space-2) 0 var(--space-3); }
      .inline-empty svg { flex-shrink: 0; }
      .net-diagram { margin: var(--space-3) 0 var(--space-4); }
      .net-hub-node { display: block; width: min(280px, 100%); margin: 0 auto; text-align: center;
                      padding: 14px 24px; background: var(--paper); border-radius: var(--radius);
                      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10); font-weight: 650;
                      font-size: 15px; color: var(--ink); }
      .net-hub-node span { display: block; font-size: 12px; font-weight: 400; color: var(--muted);
                           margin-top: 4px; }
      .net-fork { display: block; width: 100%; height: 36px; margin: 0; }
      .net-store-row { display: flex; flex-wrap: wrap; justify-content: center; gap: var(--space-2) var(--space-3); }
      .net-store-node { min-width: 88px; max-width: 140px; flex: 1 1 88px; text-align: center;
                        padding: 6px 8px; background: var(--paper); border-radius: var(--radius);
                        box-shadow: var(--shadow); font-size: 12px; color: #475569; }
      .net-store-stem { width: 1px; height: 10px; background: #cbd5e1; margin: 0 auto 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


_EMPTY_ICON = """
<svg viewBox="0 0 48 48" width="36" height="36" aria-hidden="true">
  <rect x="8" y="10" width="32" height="28" rx="4" fill="none" stroke="#94a3b8" stroke-width="2"/>
  <path d="M16 20h16M16 26h10" stroke="#94a3b8" stroke-width="2" fill="none" stroke-linecap="round"/>
</svg>
"""


def empty_state(title: str, body: str) -> None:
    render_html(
        f"""
        <div class="empty-state">
          {_EMPTY_ICON}
          <div class="empty-title">{title}</div>
          <div class="empty-body">{body}</div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    """A missing file is normal (empty_state() handles it); a *present but
    corrupt/malformed* file would otherwise raise mid-script and dump a raw
    traceback into the page. Log it and degrade the same way as missing."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        logger.exception("Failed to read %s", path)
        return None


@st.cache_data
def load_model_inputs():
    return _safe_read_csv(DATA_DIR / "model_inputs.csv")


@st.cache_data
def load_layer_a_policies():
    return _safe_read_csv(DATA_DIR / "layer_a_policies.csv")


@st.cache_data
def load_layer_b_pooling():
    return _safe_read_csv(DATA_DIR / "layer_b_pooling_results.csv")


@st.cache_data
def load_recent_demand_by_store(item_id: int, n_days: int = 30):
    df = _safe_read_csv(RAW_DIR / "train.csv", parse_dates=["date"])
    if df is None:
        return None
    try:
        sub = df[df["item"] == item_id]
        cutoff = sub["date"].max() - pd.Timedelta(days=n_days - 1)
        recent = sub[sub["date"] >= cutoff]
        return recent.pivot(index="date", columns="store", values="sales")
    except Exception:
        logger.exception("Failed to pivot demand history for item %s", item_id)
        return None


@st.cache_data
def load_pooling_static_inputs(item_id: int):
    """Sigmas, lead time, and cross-store correlation for one item - the
    service-level-independent pieces of Layer B's formula. Cached separately
    from the z-score math so the what-if slider recomputes cheaply without
    re-reading train.csv on every drag."""
    inputs = load_model_inputs()
    if inputs is None:
        return None
    item_rows = inputs[inputs["item"] == item_id].sort_values("store")
    if item_rows.empty:
        return None
    sigmas = item_rows["demand_std"].to_numpy()
    lead_time = float(item_rows["lead_time_mean_days"].mean())
    stores = item_rows["store"].to_numpy()
    corr_df = compute_correlation_matrix(item_id)
    corr_matrix = np.nan_to_num(corr_df.loc[stores, stores].to_numpy(), nan=0.0)
    return sigmas, lead_time, corr_matrix


@st.cache_data
def load_daily_sales():
    return _safe_read_csv(RAW_DIR / "train.csv", parse_dates=["date"])


@st.cache_data
def sales_date_bounds():
    sales = load_daily_sales()
    if sales is None or sales.empty:
        return None, None
    return sales["date"].min().date(), sales["date"].max().date()


@st.cache_data
def infer_on_hand_from_sales(day_iso: str) -> pd.DataFrame | None:
    """Walk a real (s,S) inventory path on train.csv sales up to `day`.

    Instant replenishment to S after a hit so the snapshot is the on-hand
    *after that day's sales* — that is the quantity Layer C uses to trigger.
    """
    policies = load_layer_a_policies()
    sales = load_daily_sales()
    if policies is None or sales is None:
        return None
    day = pd.Timestamp(day_iso)
    hist = sales[sales["date"] <= day]
    grouped = hist.groupby(["store", "item"])["sales"]

    def on_hand_after_last_sale(series, s, S):
        inv = int(S)
        vals = series.to_numpy()
        if vals.size == 0:
            return int(S)
        s = int(s)
        for q in vals[:-1]:
            inv = max(0, inv - int(q))
            if s >= 0 and inv <= s:
                inv = int(S)
        return max(0, inv - int(vals[-1]))

    rows = []
    for _, pol in policies.iterrows():
        key = (pol["store"], pol["item"])
        try:
            series = grouped.get_group(key)
        except KeyError:
            series = pd.Series(dtype=float)
        rows.append({
            "store": int(pol["store"]),
            "item": int(pol["item"]),
            "on_hand": on_hand_after_last_sale(series, pol["s"], pol["S"]),
        })
    return pd.DataFrame(rows)


@st.cache_data
def load_store_geo() -> pd.DataFrame | None:
    """10 real destination GPS points, assigned to store 1-10 (seed 42)."""
    if not LOGISTICS_PATH.exists():
        return None
    try:
        pool = get_store_coordinates()
        picked = pool.sample(n=10, random_state=42).reset_index(drop=True)
        picked.insert(0, "store", range(1, 11))
        return picked
    except Exception:
        logger.exception("Failed to read store coordinates from %s", LOGISTICS_PATH)
        return None


@st.cache_data
def load_depot_and_fleet():
    if not LOGISTICS_PATH.exists():
        return None
    try:
        df = pd.read_excel(LOGISTICS_PATH, sheet_name="Primary Data")
        origin = df[["Origin Location", "Origin Location Latitude", "Origin Location Longitude"]].dropna()
        row = origin.iloc[0]
        depot = (float(row["Origin Location Latitude"]), float(row["Origin Location Longitude"]))
        depot_name = str(row["Origin Location"])
        parsed = [parse_vehicle_capacity_tonnes(vt) for vt in df["Vehicle Type"].dropna()]
        capacities = [k for k, _ in Counter(c for c in parsed if c).most_common(6)]
        return depot, depot_name, capacities
    except Exception:
        logger.exception("Failed to read depot/fleet from %s", LOGISTICS_PATH)
        return None


@st.cache_data
def cached_vrp(requests: pd.DataFrame, store_geo: pd.DataFrame, depot: tuple,
               fleet: list, time_limit_seconds: int = 5):
    try:
        return build_vrp_model(
            requests, store_geo, depot, fleet, time_limit_seconds=time_limit_seconds
        )
    except Exception:
        logger.exception("VRP solve failed for %d requests", len(requests))
        return None


EXEC_SNAPSHOT_VRP_TIME_LIMIT = 2  # short budget: this runs on every page load, before any tab is opened


def pooling_annual_dollar_savings(pooling_df: pd.DataFrame, unit_cost: float, holding_rate: float) -> pd.Series:
    """Annual holding-cost saved per SKU from pooling safety stock at the DC
    instead of every store buffering alone: units of safety stock saved x
    unit cost x annual holding-cost rate. Same unit_cost/holding_rate for
    every row (COST_ASSUMPTIONS in data_prep.py), so a placeholder $ figure -
    directionally right, not a real per-unit cost.
    """
    return (pooling_df["ss_decentralized"] - pooling_df["ss_pooled"]) * unit_cost * holding_rate


@st.cache_data
def compute_executive_snapshot() -> dict:
    """Live KPIs for the dashboard: latest-date ROP count, pooling mean, Today's VRP.

    Uses a short VRP time budget (see EXEC_SNAPSHOT_VRP_TIME_LIMIT) since this
    runs unconditionally on every page load, before the user has chosen a tab.
    Tab 3 re-solves with a longer budget when actually opened.
    """
    pooling = load_layer_b_pooling()
    inputs = load_model_inputs()
    pooled_pct = (
        float(pooling["pct_savings"].mean())
        if pooling is not None and len(pooling)
        else None
    )
    pooled_dollars = None
    if pooling is not None and len(pooling) and inputs is not None and len(inputs):
        unit_cost = float(inputs["unit_cost_placeholder"].iloc[0])
        holding_rate = float(inputs["holding_cost_rate_annual"].iloc[0])
        pooled_dollars = float(
            pooling_annual_dollar_savings(pooling, unit_cost, holding_rate).sum()
        )
    n_stores = None
    cost = None
    dist = None
    co2 = None
    policies = load_layer_a_policies()
    _, max_d = sales_date_bounds()
    if policies is not None and max_d is not None:
        today = pd.Timestamp(max_d)
        inv = infer_on_hand_from_sales(today.strftime("%Y-%m-%d"))
        if inv is not None:
            req = build_delivery_requests(policies, inv, today)
            n_stores = int(req["store"].nunique()) if len(req) else 0
            if len(req):
                geo = load_store_geo()
                fleet_info = load_depot_and_fleet()
                if geo is not None and fleet_info is not None:
                    depot, _, fleet = fleet_info
                    result = cached_vrp(req, geo, depot, fleet, EXEC_SNAPSHOT_VRP_TIME_LIMIT)
                    if result is not None:
                        cost = float(result["total_cost"])
                        dist = float(result["total_distance_km"])
                        co2 = float(result["total_co2_kg"])
    return {
        "n_stores_below_rop": n_stores,
        "pooled_pct": pooled_pct,
        "pooled_dollars": pooled_dollars,
        "route_cost": cost,
        "route_km": dist,
        "route_co2_kg": co2,
    }


_STOCKOUT_KEYWORDS = ["stockout", "risk", "đứt hàng", "dut hang", "hết hàng", "het hang",
                      "below rop", "duoi rop", "dưới rop", "reorder"]
_TOMORROW_KEYWORDS = ["tomorrow", "ngày mai", "ngay mai"]
_SAVINGS_KEYWORDS = ["saving", "saved", "tiết kiệm", "tiet kiem", "pooled", "pooling"]
_COST_KEYWORDS = ["transport cost", "routing cost", "shipping cost", "delivery cost",
                  "chi phí vận chuyển", "chi phi van chuyen"]
_CO2_KEYWORDS = ["co2", "carbon", "emission", "khí thải", "khi thai"]


def answer_copilot_question(question: str, policies: pd.DataFrame | None,
                            inputs: pd.DataFrame | None) -> str:
    """Rule-based Q&A over this app's own precomputed data - keyword
    matching, not a real language model (no API key configured for this
    deployment). Answers a fixed set of supply-chain questions grounded in
    real numbers; anything else gets an honest "can't answer that" rather
    than a guess, since a wrong-but-confident number is worse than none."""
    q = question.lower().strip()
    if not q:
        return "Ask a question — try one of the examples below."

    if any(k in q for k in _STOCKOUT_KEYWORDS):
        if policies is None or inputs is None:
            return "Policy or model-input data isn't loaded, so I can't answer that."
        _, max_d = sales_date_bounds()
        if max_d is None:
            return "Sales history isn't loaded, so I can't answer that."
        today = pd.Timestamp(max_d)
        today_inv = infer_on_hand_from_sales(today.strftime("%Y-%m-%d"))
        if today_inv is None:
            return "Could not walk on-hand from sales history to answer that."

        use_tomorrow = any(k in q for k in _TOMORROW_KEYWORDS)
        if use_tomorrow:
            inv = project_tomorrow_on_hand(today_inv, policies, inputs)
            day_label = "tomorrow (projected)"
        else:
            inv = today_inv
            day_label = f"today ({_fmt_batch_date(today)})"

        req = build_delivery_requests(policies, inv, today)
        item_nums = re.findall(r"\d+", q)
        item_id = int(item_nums[0]) if item_nums else None
        if item_id is not None:
            req = req[req["item"] == item_id]
            subject = f"item {item_id}"
        else:
            subject = "any SKU"

        stores_at_risk = sorted(req["store"].unique().tolist())
        if not stores_at_risk:
            return f"No stores are at or below reorder point for {subject}, {day_label}."
        stores_str = ", ".join(f"Store {s}" for s in stores_at_risk)
        return f"{stores_str} — at or below reorder point for {subject}, {day_label}."

    if any(k in q for k in _SAVINGS_KEYWORDS):
        snap = compute_executive_snapshot()
        if snap["pooled_pct"] is None:
            return "Pooling results aren't loaded, so I can't answer that."
        dollars = f" (~${snap['pooled_dollars']:,.0f}/yr)" if snap["pooled_dollars"] is not None else ""
        return f"Pooling safety stock at the DC saves {snap['pooled_pct']:.1f}% on average across all SKUs{dollars}."

    if any(k in q for k in _CO2_KEYWORDS):
        snap = compute_executive_snapshot()
        if snap["route_co2_kg"] is None:
            return "No dispatch route has been solved for today, so I can't answer that."
        return f"Today's dispatch is estimated at {snap['route_co2_kg']:,.0f} kg CO2 across {snap['route_km']:.0f} km."

    if any(k in q for k in _COST_KEYWORDS):
        snap = compute_executive_snapshot()
        if snap["route_cost"] is None:
            return "No dispatch route has been solved for today, so I can't answer that."
        return f"Today's dispatch route costs ${snap['route_cost']:,.0f} across {snap['route_km']:.0f} km."

    return (
        "I can only answer a few question types right now (keyword-matched, not a "
        "real language model): stockout risk by SKU/store, pooled savings, today's "
        "routing cost, and CO2 emissions. Try rephrasing toward one of those."
    )


def _store_id_from_stop(label: str) -> int | None:
    if not label or label == "depot":
        return None
    parts = str(label).split()
    for p in parts:
        if p.isdigit():
            return int(p)
    return None


def _fmt_batch_date(d) -> str:
    return pd.Timestamp(d).strftime("%b %d, %Y")


def format_k(value: float, decimals: int = 1) -> str:
    """Compact thousands formatting for tight metric cards: 21130 -> '21.1k'."""
    if abs(value) >= 1000:
        return f"{value / 1000:.{decimals}f}k"
    return f"{value:.0f}"


def _format_eta(minutes_from_depart: float) -> str:
    """Clock time, assuming trucks roll out at DEPOT_DEPARTURE_HOUR (see
    layer_c_routing.py for why this is a documented assumption, not measured data)."""
    base = pd.Timestamp("2000-01-01") + pd.Timedelta(hours=DEPOT_DEPARTURE_HOUR)
    eta = base + pd.Timedelta(minutes=round(float(minutes_from_depart)))
    return eta.strftime("%I:%M %p").lstrip("0")


def project_tomorrow_on_hand(today_inv: pd.DataFrame, policies: pd.DataFrame,
                             inputs: pd.DataFrame) -> pd.DataFrame:
    """One more day of mean demand after today's (s,S) restock."""
    merged = today_inv.merge(policies[["store", "item", "s", "S"]], on=["store", "item"])
    means = inputs[["store", "item", "demand_mean"]]
    merged = merged.merge(means, on=["store", "item"], how="left")
    restocked = np.where(
        (merged["s"] >= 0) & (merged["on_hand"] <= merged["s"]),
        merged["S"],
        merged["on_hand"],
    )
    tomorrow = np.maximum(0, restocked - merged["demand_mean"].fillna(0)).astype(int)
    return pd.DataFrame({
        "store": merged["store"].astype(int),
        "item": merged["item"].astype(int),
        "on_hand": tomorrow,
    })


_WAREHOUSE_SVG = (
    '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">'
    '<path fill="#0c447c" d="M3 10.5 12 3l9 7.5V21H3V10.5zm2.2 1.4V19h13.6v-7.1L12 5.6 5.2 11.9z"/>'
    '<rect x="9.2" y="14.2" width="5.6" height="4.8" fill="#0c447c"/>'
    "</svg>"
)
_STORE_SVG = (
    '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">'
    '<path fill="#085041" d="M4 9.2 12 3.5l8 5.7V11H4V9.2z"/>'
    '<path fill="#085041" d="M5 12h14v8H5v-8zm3 2.2h3.2V18H8V14.2zm5.8 0H17V20h-3.2v-5.8z"/>'
    "</svg>"
)


def build_route_map(routes: list, store_geo: pd.DataFrame, depot, depot_name: str,
                    requests: pd.DataFrame) -> folium.Map:
    """Leaflet/OSM map — zoom, pan, hover. Real GPS only."""
    geo = store_geo.set_index("store")
    qty_by_store = (
        requests.groupby("store")["quantity"].sum() if len(requests) else pd.Series(dtype=int)
    )
    palette = ["#2563eb", "#dc2626", "#0f766e", "#ea580c", "#7c3aed", "#0891b2"]
    points = [(float(depot[0]), float(depot[1]))]
    for route in routes:
        for stop in route["stops"][1:-1]:
            sid = _store_id_from_stop(stop)
            if sid is None or sid not in geo.index:
                continue
            row = geo.loc[sid]
            points.append((float(row["latitude"]), float(row["longitude"])))
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    # tile.openstreetmap.org: DNS fails in this environment. CartoDB's free
    # anonymous basemaps.cartocdn.com tiles are also no longer usable - CARTO
    # now requires an account/API key and the endpoint serves a 200 OK
    # "API KEY REQUIRED" watermark tile instead of an error. Esri's
    # World_Street_Map REST tiles are verified working (curl-tested) and need
    # no key, so they're the only layer here.
    m = folium.Map(
        location=[float(np.mean(lats)), float(np.mean(lons))],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Streets",
        max_zoom=19,
    ).add_to(m)
    folium.Marker(
        location=points[0],
        tooltip=f"Depot — {depot_name}",
        popup=folium.Popup(f"<b>Depot</b><br>{depot_name}", max_width=280),
        icon=folium.Icon(color="darkblue", icon="home"),
    ).add_to(m)
    for i, route in enumerate(routes):
        color = palette[i % len(palette)]
        group = folium.FeatureGroup(name=f"Truck {route['vehicle']}")
        coords = [points[0]]
        for stop in route["stops"][1:-1]:
            sid = _store_id_from_stop(stop)
            if sid is None or sid not in geo.index:
                continue
            row = geo.loc[sid]
            lat, lon = float(row["latitude"]), float(row["longitude"])
            coords.append((lat, lon))
            extra = f" {stop[stop.find('('):]}" if "(" in str(stop) else ""
            qty = int(qty_by_store.get(sid, 0))
            label = (
                f"<b>Store {sid}{extra}</b><br>{row['location_name']}"
                f"<br>Drop off: {qty:,} units"
            )
            folium.CircleMarker(
                location=(lat, lon),
                radius=8,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                tooltip=label,
                popup=folium.Popup(label, max_width=280),
            ).add_to(group)
        coords.append(points[0])
        folium.PolyLine(coords, color=color, weight=4, opacity=0.85).add_to(group)
        group.add_to(m)
    # fit_bounds() run synchronously here computes the zoom from the map
    # container's *current* pixel size, before the Streamlit iframe has
    # settled to its final size - verified via a live run: zoom landed on
    # maxZoom (19), showing a ~150m sliver around the bounds centroid instead
    # of the intended country-wide view. Deferring the same call to after
    # layout settles (paired with the resize dispatch below) fixes it.
    sw = [min(lats), min(lons)]
    ne = [max(lats), max(lons)]
    m.get_root().html.add_child(folium.Element(
        f"""
        <script>
        setTimeout(function() {{
            {m.get_name()}.invalidateSize();
            {m.get_name()}.fitBounds([{sw}, {ne}], {{padding: [40, 40]}});
        }}, 300);
        </script>
        """
    ))
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def render_route_map(m: folium.Map, height: int = 520) -> None:
    """Embed Leaflet in Streamlit. Bootstrap's img { max-width:100% } otherwise hides tiles."""
    m.get_root().header.add_child(folium.Element(
        """
        <style>
          .leaflet-container img.leaflet-tile {
            max-width: none !important;
            max-height: none !important;
          }
        </style>
        """
    ))
    m.get_root().html.add_child(folium.Element(
        "<script>setTimeout(function(){window.dispatchEvent(new Event('resize'));}, 250);</script>"
    ))
    fig = folium.Figure(width="100%", height=height)
    fig.add_child(m)
    st.components.v1.html(fig.render(), height=height + 8)


def stop_timeline_html(route: dict, store_geo: pd.DataFrame, depot_name: str,
                       stops_df: pd.DataFrame) -> str:
    geo = store_geo.set_index("store")
    rows = stops_df[stops_df["vehicle"] == route["vehicle"]].sort_values("seq")
    parts = ['<div style="padding-left:4px;">']
    n = len(rows)
    for i, row in enumerate(rows.itertuples()):
        is_depot = str(row.stop) == "depot"
        if is_depot:
            bg, icon = "#93c5fd", _WAREHOUSE_SVG
            title = "Return to depot" if i == n - 1 and n > 1 else "Depot"
            meta = depot_name
            drop = ""
        else:
            bg, icon = "#5eead4", _STORE_SVG
            sid = _store_id_from_stop(row.stop)
            loc = geo.loc[sid, "location_name"] if sid in geo.index else row.stop
            extra = f" {str(row.stop)[str(row.stop).find('('):]}" if "(" in str(row.stop) else ""
            title = f"Store {sid}{extra}"
            meta = loc
            units = int(round(row.demand_kg / KG_PER_UNIT))
            drop = (
                '<div style="display:inline-block;margin-top:4px;background:#eff6ff;'
                'color:#1d4ed8;font-size:12px;font-weight:600;padding:2px 8px;'
                f'border-radius:999px;">Drop off: {units:,} units</div>'
                if units else ""
            )
        line = (
            '<div style="width:2px;flex:1;min-height:18px;background:#e2e8f0;"></div>'
            if i < n - 1 else ""
        )
        eta_str = _format_eta(row.eta_minutes)
        parts.append(
            f'<div style="display:flex;gap:12px;align-items:flex-start;">'
            f'<div style="width:28px;display:flex;flex-direction:column;'
            f'align-items:center;flex-shrink:0;">'
            f'<div style="width:28px;height:28px;border-radius:50%;background:{bg};'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            f'{icon}</div>{line}</div>'
            f'<div style="padding-bottom:14px;">'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<div style="font-weight:650;font-size:14px;color:#0f172a;">{title}</div>'
            f'<div style="font-size:12px;font-weight:600;color:#334155;">ETA {eta_str}</div>'
            f'</div>'
            f'<div style="font-size:12px;color:#64748b;">{meta}</div>{drop}'
            f'</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def build_driver_manifest(stops_df: pd.DataFrame, store_geo: pd.DataFrame) -> pd.DataFrame:
    """Flatten the VRP stop sequence into a print/CSV sheet a driver can follow."""
    geo = store_geo.set_index("store")
    rows = []
    for r in stops_df.itertuples():
        sid = _store_id_from_stop(r.stop)
        if sid is None:
            location = "—"
        elif sid in geo.index:
            location = geo.loc[sid, "location_name"]
        else:
            location = ""
        rows.append({
            "truck": r.vehicle,
            "stop_seq": r.seq,
            "stop": r.stop,
            "location": location,
            "units_to_deliver": int(round(r.demand_kg / KG_PER_UNIT)),
            "cumulative_load_kg": r.cumul_load_kg,
            "eta": _format_eta(r.eta_minutes),
        })
    return pd.DataFrame(rows)


def erp_inventory_snapshot(store, item, s, S, demand_mean, demand_std):
    """Deterministic on-hand + open-PO pull, seeded per (store, SKU).

    On-hand tracks ~1 day of demand so it sits far below S (the tank scrapes
    the bottom). ~40% of low-stock SKUs already have an (s,S) replenishment
    in transit — that faded layer is what blocks a second order.
    """
    rng = np.random.default_rng(10_000 + int(store) * 100 + int(item))
    s = max(int(s), 0)
    S = max(int(S), 1)
    mu = float(demand_mean)
    sigma = max(float(demand_std), 1.0)

    if rng.random() < 0.72:
        on_hand = int(np.clip(rng.normal(mu, sigma * 0.25), 1, max(s - 1, 1)))
        in_transit = int(S - on_hand) if rng.random() < 0.42 else 0
    else:
        on_hand = int(np.clip(rng.uniform(s + 1, max(s + 1, S * 0.45)), s + 1, S))
        in_transit = 0

    on_hand = int(np.clip(on_hand, 0, S))
    in_transit = int(np.clip(in_transit, 0, S - on_hand))
    return on_hand, in_transit


def build_store_po(store: int, model_inputs_df: pd.DataFrame, policies_df: pd.DataFrame) -> pd.DataFrame:
    """Every SKU at this store below ROP with no PO already in transit - the
    same simulated ERP snapshot the tank gauge uses, reused so the export
    matches what tab 1 shows on screen."""
    rows = []
    for _, row in model_inputs_df[model_inputs_df["store"] == store].iterrows():
        pol = policies_df[(policies_df["store"] == store) & (policies_df["item"] == row["item"])]
        if pol.empty:
            continue
        pol = pol.iloc[0]
        s_i, S_i = max(int(pol["s"]), 0), max(int(pol["S"]), 1)
        on_hand, in_transit = erp_inventory_snapshot(
            store, row["item"], s_i, S_i, row["demand_mean"], row["demand_std"]
        )
        if on_hand <= s_i and in_transit == 0:
            rows.append({
                "store": store, "item": int(row["item"]), "on_hand": on_hand,
                "reorder_point_s": s_i, "order_up_to_S": S_i, "order_qty": S_i - on_hand,
            })
    return pd.DataFrame(rows, columns=[
        "store", "item", "on_hand", "reorder_point_s", "order_up_to_S", "order_qty"
    ])


def sparkline_fig(values, color: str, *, smooth: bool = False, height: int = 96, dates=None,
                   forecast: pd.Series | None = None):
    """Compact demand sparkline that fills the card; axes hidden but hover +
    min/max end-labels give a reader the actual numbers behind the shape.

    `forecast`, if given, is a dated Series continuing past `dates`' last
    point - drawn as a dashed continuation of the same line, not a new color,
    so it reads as "this line, projected" rather than a second series.
    """
    y = np.asarray(values, dtype=float)
    if smooth and len(y) >= 5:
        y = pd.Series(y).rolling(5, min_periods=1, center=True).mean().to_numpy()
    x = pd.to_datetime(dates) if dates is not None else np.arange(len(y))
    hovertemplate = (
        "%{x|%b %d}: %{y:.0f} units<extra></extra>"
        if dates is not None
        else "%{y:.0f} units<extra></extra>"
    )
    # shape="spline" softens the line's corners for rendering only (still
    # passes through every real data point, just no longer sharp-angled) -
    # `smooth` above is a separate, data-level rolling mean.
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2.4, shape="spline", smoothing=0.6),
            hovertemplate=hovertemplate,
            name="actual",
        )
    )
    all_y = [y]
    if forecast is not None and len(forecast) and dates is not None:
        fx = pd.to_datetime([x[-1]] + list(forecast.index))
        fy = np.clip(np.asarray([y[-1]] + list(forecast.to_numpy(dtype=float))), 0, None)
        fig.add_trace(go.Scatter(
            x=fx,
            y=fy,
            mode="lines",
            line=dict(color=color, width=2.0, dash="dash", shape="spline", smoothing=0.6),
            opacity=0.6,
            hovertemplate="%{x|%b %d}: %{y:.0f} units (forecast)<extra></extra>",
            name="forecast",
        ))
        all_y.append(fy)
    y_min, y_max = float(np.min(np.concatenate(all_y))), float(np.max(np.concatenate(all_y)))
    label_style = dict(showarrow=False, xref="paper", yref="y", xanchor="right",
                        xshift=-4, font=dict(size=10, color="#94a3b8"))
    fig.add_annotation(x=0, y=y_max, text=f"{y_max:.0f}", yanchor="top", **label_style)
    fig.add_annotation(x=0, y=y_min, text=f"{y_min:.0f}", yanchor="bottom", **label_style)
    fig.update_layout(
        height=height,
        margin=dict(l=28, r=4, t=6, b=4),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        autosize=True,
        hoverlabel=dict(font_size=11),
    )
    return fig


def volatility_style(series) -> tuple[str, str]:
    """Map demand roughness to an ops-friendly color + label (no CV)."""
    mean = float(series.mean())
    cv = float(series.std() / mean) if mean else 0.0
    if cv >= 0.45:
        return "#dc2626", "Erratic"
    if cv >= 0.30:
        return "#ea580c", "Uneven"
    return "#64748b", "Steady"


def show_sparkline(values, color: str, key: str, *, smooth: bool = False, height: int = 96, dates=None,
                    forecast: pd.Series | None = None):
    # staticPlot=True previously blocked hover entirely regardless of the
    # figure's own hover settings - dropped so the tooltip actually works.
    st.plotly_chart(
        sparkline_fig(values, color, smooth=smooth, height=height, dates=dates, forecast=forecast),
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        key=key,
    )


model_inputs = load_model_inputs()
policies_precomputed = load_layer_a_policies()
pooling_precomputed = load_layer_b_pooling()

with st.sidebar:
    st.markdown("### Inventory Control")
    _, max_d = sales_date_bounds()
    st.caption(
        f"Data as of **{_fmt_batch_date(max_d)}** — latest date in this historical "
        "dataset (a frozen Kaggle export, not a live feed)."
        if max_d is not None else "Data as of —"
    )

    st.markdown("##### Data sources")
    for label, ok in [
        ("Demand history (train.csv)", load_daily_sales() is not None),
        ("Layer A policies", policies_precomputed is not None),
        ("Layer B pooling results", pooling_precomputed is not None),
        ("Logistics / GPS dataset", LOGISTICS_PATH.exists()),
    ]:
        mark = (
            '<span style="color:#0f766e;font-weight:700;">✓</span>' if ok
            else '<span style="color:#dc2626;font-weight:700;">✗</span>'
        )
        render_html(f'<div style="font-size:13px;color:#475569;margin:2px 0;">{mark} {label}</div>')

    with st.expander("Quick guide"):
        st.markdown(
            "**Replenishment Strategy** — pick a store + SKU to see its (s, S) reorder "
            "policy and current stock gauge.\n\n"
            "**Risk Pooling** — compare safety stock held per-store vs. pooled centrally "
            "per SKU, plus a what-if service-level slider.\n\n"
            "**Dispatch Routes** — today's delivery routes for stores below reorder "
            "point, solved as a capacitated VRP with ETAs."
        )

    st.markdown("##### Supply Chain Copilot")
    st.caption(
        "Keyword-matched Q&A over this app's own data — not a live language "
        "model (no API key configured for this deployment)."
    )
    st.caption(
        'Try: "Which stores risk a stockout on item 12 tomorrow?" · '
        '"How much are we saving from pooling?" · "What\'s today\'s routing cost?"'
    )
    copilot_q = st.text_input(
        "Ask a question", key="copilot_q", label_visibility="collapsed",
        placeholder="Ask about stockouts, savings, cost, CO2…",
    )
    if copilot_q:
        st.info(answer_copilot_question(copilot_q, policies_precomputed, model_inputs))

st.title("Inventory Control")
st.caption("10 stores × 50 SKUs · (s, S) replenishment, risk pooling, and capacitated dispatch")

render_html(
    """
    <div class="pillar-row">
      <div class="pillar-card">
        <h3>Replenishment</h3>
        <p>Per store–SKU (s, S) policy from Bellman value iteration. Reorder at s, raise on-hand to S.</p>
      </div>
      <div class="pillar-card">
        <h3>Risk pooling</h3>
        <p>Safety stock if each store buffers alone versus pooling demand at the DC.</p>
      </div>
      <div class="pillar-card">
        <h3>Dispatch</h3>
        <p>Capacitated VRP for stores at or below reorder point on the latest sales date.</p>
      </div>
    </div>
    """
)

st.markdown("## Executive Dashboard")
snap = compute_executive_snapshot()
dash1, dash2, dash3 = st.columns(3)
rop_val = snap["n_stores_below_rop"]
dash1.metric(
    "Stores Below ROP",
    "—" if rop_val is None else f"{int(rop_val)}",
    help="Stores with at least one SKU at or below reorder point s on the latest sales date (labeled Today). Walked from train.csv under the (s, S) policy.",
)
dash2.metric(
    "Total Pooled Savings (%)",
    "—" if snap["pooled_pct"] is None else f"{snap['pooled_pct']:.1f}%",
    delta=None if snap["pooled_dollars"] is None else f"${snap['pooled_dollars']:,.0f}/yr",
    delta_color="off",
    help="Mean percent reduction in safety stock from pooling at the DC versus holding a buffer at every store, averaged across all SKUs. Dollar figure is the annual holding-cost saved across all 50 SKUs (units of safety stock saved x unit_cost_placeholder x holding_cost_rate_annual) - directionally right, not a real per-unit cost.",
)
if snap["route_cost"] is None:
    dash3.metric(
        "Total Routing Cost/Distance",
        "—",
        help="Today's capacitated VRP: transport cost ($1.50/km + $50 per used truck) and total haversine km. Empty when no SKUs are below ROP or routing inputs are missing.",
    )
else:
    dash3.metric(
        "Total Routing Cost/Distance",
        f"${snap['route_cost']:,.0f}",
        delta=f"{snap['route_km']:.0f} km",
        delta_color="off",
        help="Today's capacitated VRP: transport cost ($1.50/km + $50 per used truck) shown as the value; haversine kilometers on real logistics GPS shown underneath.",
    )

tab1, tab2, tab3 = st.tabs(["Replenishment Strategy", "Risk Pooling", "Dispatch Routes"])


# ===========================================================================
# TAB 1 - Replenishment Strategy (Layer A)
# ===========================================================================
with tab1:
    if model_inputs is None:
        empty_state(
            "Model inputs not found",
            "Run notebooks/01_data_preparation.ipynb to write data/processed/model_inputs.csv.",
        )
    else:
        col_select_a, col_select_b = st.columns(2)
        with col_select_a:
            store = st.selectbox(
                "Store",
                sorted(model_inputs["store"].unique()),
                key="t1_store",
                help="Store ID from the sales file (1–10).",
            )
        with col_select_b:
            items_for_store = sorted(model_inputs.loc[model_inputs["store"] == store, "item"].unique())
            item = st.selectbox(
                "SKU",
                items_for_store,
                key="t1_item",
                help="Item ID from the sales file (1–50).",
            )

        row = model_inputs[(model_inputs["store"] == store) & (model_inputs["item"] == item)]
        if row.empty:
            empty_state(
                "No rows for this store–SKU",
                "model_inputs.csv has no matching pair. Pick another store or SKU.",
            )
        else:
            row = row.iloc[0]

            precomputed_row = None
            if policies_precomputed is not None:
                match = policies_precomputed[
                    (policies_precomputed["store"] == store) & (policies_precomputed["item"] == item)
                ]
                if not match.empty:
                    precomputed_row = match.iloc[0]

            if precomputed_row is not None:
                s_val, S_val = int(precomputed_row["s"]), int(precomputed_row["S"])
            else:
                with st.spinner("Solving policy…"):
                    result = solve_for_row(row)
                s_val, S_val = int(result["s"]), int(result["S"])

            S_val = max(S_val, 1)
            s_plot = max(s_val, 0)

            on_hand, in_transit = erp_inventory_snapshot(
                store, item, s_plot, S_val, row["demand_mean"], row["demand_std"]
            )
            inventory_position = on_hand + in_transit

            # Dynamic ROP: a simple heuristic overlay on Layer A's static (s, S) -
            # not a re-solve of the Bellman equation. If a short-horizon forecast
            # (src/demand_forecast.py) says demand over the lead time will run
            # hotter than the historical mean the DP assumed, temporarily raise
            # the trigger threshold by that excess so a forecasted spike shows up
            # before on-hand actually crashes through the static s.
            lead_time_days = max(float(row["lead_time_mean_days"]), 1.0)
            horizon = max(int(round(lead_time_days)), 1)
            demand_hist = load_recent_demand_by_store(item, n_days=30)
            hist_series = (
                demand_hist[store] if demand_hist is not None and store in demand_hist.columns else None
            )
            s_effective = s_plot
            dynamic_note = None
            demand_forecast = pd.Series(dtype=float)
            if hist_series is not None:
                demand_forecast = forecast_demand(hist_series, horizon=horizon)
                fc = demand_forecast
                if len(fc):
                    forecast_total = float(fc.sum())
                    baseline_total = float(row["demand_mean"]) * horizon
                    excess = forecast_total - baseline_total
                    if excess > 0:
                        s_effective = min(s_plot + int(round(excess)), S_val - 1)
                        dynamic_note = (
                            f"Demand forecast over the next {horizon}-day lead time is "
                            f"{forecast_total:.0f} units, {excess:.0f} above the "
                            f"{baseline_total:.0f}-unit baseline the static policy assumed — "
                            f"reorder point raised from {s_plot} to {s_effective} until the forecast eases."
                        )

            on_hand_pct = (on_hand / S_val) * 100
            transit_pct = (in_transit / S_val) * 100
            s_pct = (s_effective / S_val) * 100

            if on_hand > s_effective:
                badge_class, badge_text = "status-ok", f"Current Stock: {on_hand} units - Healthy"
            elif in_transit > 0:
                badge_class, badge_text = (
                    "status-transit",
                    f"Current Stock: {on_hand} units - {in_transit} in transit",
                )
            else:
                badge_class, badge_text = (
                    "status-urgent",
                    f"Current Stock: {on_hand} units - Urgent Restock!",
                )

            render_html(f'<div class="status-badge {badge_class}">{badge_text}</div>')
            st.caption(
                "On-hand / in-transit here are a simulated ERP snapshot for this store–SKU "
                "(seeded, not a live figure) — used to demo the reorder gauge. "
                "Dispatch Routes (Tab 3) instead walks real on-hand from actual sales history, "
                "so the two tabs can show different numbers for the same store/SKU/day by design."
            )
            st.markdown("")

            col_tank, col_info = st.columns([0.6, 2.4])

            with col_tank:
                tank_html = f"""
                <div style="display:flex; justify-content:center; align-items:flex-end; gap:10px;">
                  <div style="position:relative; height:220px; width:58px; font-size:11px; color:#475569; font-weight:600; text-align:right;">
                    <div style="position:absolute; top:0; right:0;">S {S_val}</div>
                    <div style="position:absolute; bottom:{s_pct:.2f}%; right:0; transform:translateY(50%);">ROP {s_effective}</div>
                    <div style="position:absolute; bottom:0; right:0;">0</div>
                  </div>
                  <div style="position:relative; width:88px; height:220px;
                              filter: drop-shadow(0 4px 10px rgba(15, 23, 42, 0.12));">
                    <div style="position:absolute; inset:0; border:1px solid #cbd5e1; border-radius:8px;
                                background:#f1f5f9; overflow:hidden;">
                      <div style="position:absolute; bottom:0; left:0; right:0; height:{on_hand_pct:.2f}%;
                                  background: linear-gradient(180deg, #3b82f6 0%, #1e3a5f 100%);"></div>
                      <div style="position:absolute; bottom:{on_hand_pct:.2f}%; left:0; right:0;
                                  height:{transit_pct:.2f}%;
                                  background: repeating-linear-gradient(-45deg,
                                    rgba(147,197,253,0.7),
                                    rgba(147,197,253,0.7) 7px,
                                    rgba(219,234,254,0.45) 7px,
                                    rgba(219,234,254,0.45) 14px);"></div>
                      <div style="position:absolute; left:0; right:0; bottom:{s_pct:.2f}%;
                                  border-top:2px dashed #c2410c;"></div>
                      <div style="position:absolute; left:0; right:0; top:0;
                                  border-top:2px dashed #1e3a5f;"></div>
                    </div>
                  </div>
                </div>
                <div class="tank-legend" style="text-align:center;">
                  <div><span style="background:linear-gradient(180deg,#3b82f6,#1e3a5f);"></span>On-hand</div>
                  <div><span style="background:repeating-linear-gradient(-45deg,#93c5fd,#93c5fd 4px,#dbeafe 4px,#dbeafe 8px);
                                    border:1px solid #93c5fd;"></span>In-transit</div>
                </div>
                """
                render_html(tank_html)

            with col_info:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Reorder Point",
                    f"{s_effective}",
                    delta=f"+{s_effective - s_plot} forecast" if s_effective > s_plot else None,
                    delta_color="inverse",
                    help="Static s from Layer A's Bellman value iteration, dynamically raised when a "
                         "short-horizon demand forecast exceeds the historical mean the DP assumed "
                         "(see the note below the metrics).",
                )
                m2.metric(
                    "Order-up-to Level",
                    f"{S_val}",
                    help="S in the (s, S) policy. Target on-hand after a replenishment. Solved together with s by Bellman value iteration.",
                )
                m3.metric(
                    "In-transit",
                    f"{in_transit}",
                    help="Units already on an open purchase order. Added to on-hand to form inventory position so this screen does not raise a second order.",
                )

                if dynamic_note:
                    render_html(
                        f'<div class="ops-note" style="border-left:3px solid #d97706;">{dynamic_note}</div>'
                    )

                if on_hand <= s_effective and in_transit == 0:
                    need = S_val - on_hand
                    render_html(
                        f'<div class="ops-note">Below ROP. Raise {need} units to order-up-to {S_val}.</div>'
                    )
                elif on_hand <= s_effective and inventory_position >= s_effective:
                    render_html(
                        f'<div class="ops-note">Open PO covers the gap. Inventory position '
                        f"{inventory_position} (on-hand {on_hand} + in-transit {in_transit}). "
                        "Do not double-order.</div>"
                    )
                else:
                    render_html('<div class="ops-note">Above ROP. No replenishment.</div>')

            po_df = build_store_po(store, model_inputs, policies_precomputed)
            if not po_df.empty:
                st.markdown("")
                st.download_button(
                    f"Download Purchase Order — Store {store} ({len(po_df)} SKUs)",
                    data=po_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"po_store_{store}.csv",
                    mime="text/csv",
                    type="primary",
                    key="po_download",
                    help="Every SKU at this store below reorder point with no PO already in transit.",
                )

            if hist_series is not None:
                st.markdown("")
                st.markdown("##### Recent demand — this store & SKU (last 30 days)")
                hist_color, hist_label = volatility_style(hist_series)
                forecast_suffix = (
                    f" · dashed tail is the {horizon}-day forecast driving the reorder point above"
                    if len(demand_forecast) else ""
                )
                st.caption(f"{hist_label} demand — context for why the policy above landed where it did.{forecast_suffix}")
                show_sparkline(
                    hist_series.to_numpy(), hist_color, key="spark_tab1_demand",
                    smooth=False, height=120, dates=hist_series.index, forecast=demand_forecast,
                )


# ===========================================================================
# TAB 2 - Risk Pooling (Layer B)
# ===========================================================================
with tab2:
    if pooling_precomputed is None:
        empty_state(
            "Pooling results not found",
            "Run notebooks/03_layer_b_risk_pooling.ipynb to write data/processed/layer_b_pooling_results.csv.",
        )
    else:
        item_for_network = st.selectbox(
            "SKU",
            sorted(pooling_precomputed["item"].unique()),
            key="t2_item",
            help="Item ID from the sales file. Pooling math is computed per SKU across stores.",
        )

        item_row = pooling_precomputed[pooling_precomputed["item"] == item_for_network].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Store-level safety stock",
                f"{item_row['ss_decentralized']:.0f}",
                help="Sum of per-store safety stock if each store holds its own buffer: z × σᵢ × √L. No pooling.",
            )
        with col2:
            st.metric(
                "Pooled at DC",
                f"{item_row['ss_pooled']:.0f}",
                delta=f"-{item_row['pct_savings']:.1f}%",
                delta_color="inverse",
                help="Safety stock if demand is pooled at the DC: z × √(Σᵢ Σⱼ σᵢ σⱼ ρᵢⱼ) × √L. Delta is (ss_decentralized − ss_pooled) / ss_decentralized.",
            )
        with col3:
            if model_inputs is not None and len(model_inputs):
                unit_cost = float(model_inputs["unit_cost_placeholder"].iloc[0])
                holding_rate = float(model_inputs["holding_cost_rate_annual"].iloc[0])
                dollar_saved = (item_row["ss_decentralized"] - item_row["ss_pooled"]) * unit_cost * holding_rate
                st.metric(
                    "Est. annual $ saved",
                    f"${dollar_saved:,.0f}",
                    help="(Store-level − Pooled) safety stock x unit_cost_placeholder x holding_cost_rate_annual for this SKU. Placeholder unit cost, so directionally right, not a real dollar figure.",
                )

        with st.container(border=True):
            st.markdown("##### What-if: target service level")
            st.caption(
                "Recomputes this SKU's safety stock at a different service level with Layer B's own "
                "closed-form formula (z from the service level, same σ and cross-store correlation as above). "
                "Layer A's (s, S) policy on Tab 1 is cost-based, not parameterized by a fill-rate target, "
                "so it does not move with this slider."
            )
            service_level_pct = st.slider(
                "Target service level", min_value=80.0, max_value=99.9, value=95.0,
                step=0.5, format="%.1f%%", key="t2_service_level",
            )
            service_level = service_level_pct / 100.0
            static_inputs = load_pooling_static_inputs(item_for_network)
            if static_inputs is not None:
                sigmas, lead_time, corr_matrix = static_inputs
                z = float(norm.ppf(service_level))
                ss_dec_wi = decentralized_safety_stock(sigmas, z, lead_time)
                ss_pool_wi = pooled_safety_stock(sigmas, corr_matrix, z, lead_time)
                wi1, wi2 = st.columns(2)
                wi1.metric("Store-level safety stock (what-if)", f"{ss_dec_wi:.0f}")
                wi_pct = 100 * (ss_dec_wi - ss_pool_wi) / ss_dec_wi if ss_dec_wi else 0.0
                wi2.metric(
                    "Pooled at DC (what-if)", f"{ss_pool_wi:.0f}",
                    delta=f"-{wi_pct:.1f}%", delta_color="inverse",
                )

        demand_by_store = load_recent_demand_by_store(item_for_network)
        stores = (
            sorted(demand_by_store.columns)
            if demand_by_store is not None
            else list(range(1, 11))
        )

        st.markdown("### Demand volatility (last 30 days)")
        render_html(
            """
            <div class="demand-legend">
              <b style="background:#1e3a5f;"></b>DC (smoothed)
              <b style="background:#64748b;"></b>Steady
              <b style="background:#ea580c;"></b>Uneven
              <b style="background:#dc2626;"></b>Erratic
            </div>
            """
        )

        hub_values = None
        hub_forecast = None
        if demand_by_store is not None:
            hub_series = demand_by_store.mean(axis=1)
            hub_values = hub_series.to_numpy()
            hub_forecast = forecast_demand(hub_series, horizon=7)

        store_nodes = "".join(
            f'<div class="net-store-node"><div class="net-store-stem"></div>'
            f"{BRANCH_NAMES.get(int(s_id), f'Store {int(s_id)}')}</div>"
            for s_id in stores
        )
        render_html(
            f"""
            <div class="net-diagram">
              <div class="net-hub-node">Central Warehouse
                <span>Pooled demand hub</span>
              </div>
              <svg class="net-fork" viewBox="0 0 400 36" preserveAspectRatio="none" aria-hidden="true">
                <path d="M200 0 V14" stroke="#cbd5e1" stroke-width="1.5" fill="none"/>
                <path d="M24 14 H376" stroke="#cbd5e1" stroke-width="1.5" fill="none"/>
                <path d="M24 14 V36 M80 14 V36 M136 14 V36 M192 14 V36 M248 14 V36 M304 14 V36 M360 14 V36 M376 14 V36"
                      stroke="#e2e8f0" stroke-width="1.25" fill="none"/>
              </svg>
              <div class="net-store-row">{store_nodes}</div>
            </div>
            """
        )

        with st.container(border=True):
            icon_col, title_col = st.columns([0.07, 0.93])
            with icon_col:
                render_html(WAREHOUSE_ICON)
            with title_col:
                st.markdown("**Central Warehouse**")
                st.caption("Pooled demand — smoothed · dashed tail is a 7-day forecast (Holt-Winters)")
            if hub_values is not None:
                show_sparkline(hub_values, "#1e3a5f", key="spark_hub", smooth=True, height=160,
                               dates=demand_by_store.index, forecast=hub_forecast)

        def render_store_card(s_id: int) -> None:
            color, label = "#64748b", "No data"
            series = None
            if demand_by_store is not None and s_id in demand_by_store.columns:
                series = demand_by_store[s_id]
                color, label = volatility_style(series)
            name = BRANCH_NAMES.get(int(s_id), f"Store {s_id}")
            with st.container(border=True):
                icon_col, title_col = st.columns([0.22, 0.78])
                with icon_col:
                    render_html(STORE_ICON)
                with title_col:
                    st.markdown(f"**{name}**")
                    st.caption(label)
                if series is not None:
                    show_sparkline(
                        series.to_numpy(),
                        color,
                        key=f"spark_store_{int(s_id)}",
                        smooth=False,
                        height=100,
                        dates=series.index,
                    )

        for chunk_start in range(0, len(stores), 5):
            chunk = stores[chunk_start : chunk_start + 5]
            cols = st.columns(len(chunk))
            for col, s_id in zip(cols, chunk):
                with col:
                    render_store_card(s_id)


# ===========================================================================
# TAB 3 - Dispatch Routes (Layer C)
# ===========================================================================
with tab3:
    if policies_precomputed is None:
        st.info("Run notebooks/02_layer_a_dp_inventory.ipynb to build (s, S) policies first.")
    else:
        min_d, max_d = sales_date_bounds()
        if min_d is None:
            st.warning("train.csv is missing — real daily sales are required.")
        else:
            today = pd.Timestamp(max_d)
            yesterday = today - pd.Timedelta(days=1)
            today_label = _fmt_batch_date(today)
            yesterday_label = _fmt_batch_date(yesterday)

            today_inv = infer_on_hand_from_sales(today.strftime("%Y-%m-%d"))
            if today_inv is None:
                st.warning("Could not walk on-hand from train.csv for this date.")
            else:
                today_req = build_delivery_requests(policies_precomputed, today_inv, today)

                batch_options = [
                    f"Pending for Today ({today_label})",
                    "Scheduled for Tomorrow",
                    f"Past Dispatches ({yesterday_label})",
                ]
                default_ix = 0 if len(today_req) else 2
                batch = st.selectbox(
                    "Select dispatch batch",
                    batch_options,
                    index=default_ix,
                    key="t3_batch",
                    help="Today is the latest date in train.csv. Tomorrow projects one more day of mean demand after (s, S) restock. Past is the previous sales day.",
                )

                if batch.startswith("Pending"):
                    inventory, requests = today_inv, today_req
                elif batch.startswith("Scheduled"):
                    day_ts = today + pd.Timedelta(days=1)
                    inventory = project_tomorrow_on_hand(
                        today_inv, policies_precomputed, model_inputs
                    )
                    requests = build_delivery_requests(
                        policies_precomputed, inventory, day_ts
                    )
                else:
                    inventory = infer_on_hand_from_sales(yesterday.strftime("%Y-%m-%d"))
                    if inventory is None:
                        st.warning("Could not walk on-hand for the previous sales day.")
                        requests = pd.DataFrame(
                            columns=["store", "item", "quantity", "requested_date"]
                        )
                    else:
                        requests = build_delivery_requests(
                            policies_precomputed, inventory, yesterday
                        )

                store_geo = load_store_geo()
                fleet_info = load_depot_and_fleet()
                if store_geo is None or fleet_info is None:
                    st.warning(
                        "Logistics dataset not found at "
                        f"{LOGISTICS_PATH.relative_to(PROJECT_ROOT)} — dispatch routing needs it "
                        "for depot/fleet/store GPS."
                    )
                    st.stop()
                depot, depot_name, fleet = fleet_info
                n_stores = int(requests["store"].nunique()) if len(requests) else 0
                store_word = "store" if n_stores == 1 else "stores"

                if n_stores > 0:
                    render_html(
                        '<div style="background:#fef2f2;border:1px solid #fecaca;'
                        'border-radius:10px;padding:12px 16px;color:#991b1b;'
                        f'font-weight:600;margin-bottom:12px;">Action Required: '
                        f"{n_stores} {store_word} are below Reorder Point. "
                        "Generating optimal dispatch route…</div>"
                    )
                else:
                    render_html(
                        f"""
                        <div class="inline-empty">
                          {_EMPTY_ICON}
                          <span>No dispatch route for this batch — no SKUs are below reorder point.</span>
                        </div>
                        """
                    )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(
                    "Stores",
                    f"{n_stores}",
                    help="Stores with at least one SKU at or below reorder point s for the selected batch.",
                )
                m2.metric(
                    "SKUs",
                    f"{len(requests)}",
                    help="Store–SKU pairs whose walked on-hand is ≤ reorder point s. Order quantity = S − on-hand.",
                )

                if requests.empty:
                    m3.metric(
                        "Distance",
                        "—",
                        help="Haversine km on real logistics GPS once a route is solved. Empty when this batch has no requests.",
                    )
                    m4.metric(
                        "Cost",
                        "—",
                        help="Capacitated VRP cost ($1.50/km + $50 per used truck) once a route is solved. Empty when this batch has no requests.",
                    )
                    m5.metric(
                        "CO2",
                        "—",
                        help="Estimated at 0.10 kg CO2 per tonne-km carried, once a route is solved. Empty when this batch has no requests.",
                    )
                else:
                    with st.spinner("Building dispatch routes…"):
                        result = cached_vrp(
                            requests, store_geo, depot, fleet, 5
                        )
                    if result is None:
                        st.error(
                            "Could not build a dispatch route for this batch — the routing "
                            "solver hit an unexpected error. Try a different batch, or check "
                            "the logs for details."
                        )
                        st.stop()

                    m3.metric(
                        "Distance",
                        f"{format_k(result['total_distance_km'])} km",
                        help="Haversine km on real logistics GPS (Destination Location, seed=42) — not Kaggle store addresses.",
                    )
                    m4.metric(
                        "Cost",
                        f"${format_k(result['total_cost'])}",
                        help="Capacitated VRP (OR-Tools, PATH_CHEAPEST_ARC): $1.50/km + $50 per used truck. Feedback signal for Layer A’s K.",
                    )
                    m5.metric(
                        "CO2",
                        f"{format_k(result['total_co2_kg'])} kg",
                        help="0.10 kg CO2 per tonne-km carried (documented estimate — no fuel data in either "
                             "source dataset), applied to each route's actual load and distance.",
                    )
                    st.caption(
                        f"Today in this dataset is **{today_label}** (latest sales date). "
                        f"Depot: {depot_name} ({depot[0]:.4f}, {depot[1]:.4f})."
                    )

                    st.markdown("#### Stop sequence")
                    st.caption(
                        f"ETA assumes trucks depart the depot at {DEPOT_DEPARTURE_HOUR}:00, "
                        f"{AVG_TRUCK_SPEED_KMH:.0f} km/h average, {STOP_SERVICE_MINUTES:.0f} min at each stop "
                        "to unload — no timing data exists in either source dataset, so treat as an estimate."
                    )
                    for route in result["routes"]:
                        with st.container(border=True):
                            st.markdown(
                                f"**Truck {route['vehicle']}** · {route['distance_km']} km · "
                                f"{route['load_kg']/1000:.1f} / {route['capacity_kg']/1000:.1f} t loaded · "
                                f"{route['co2_kg']:.0f} kg CO2"
                            )
                            render_html(stop_timeline_html(
                                route, store_geo, depot_name, result["stops"]
                            ))

                    st.download_button(
                        "Download Driver Route Sheet (CSV)",
                        data=build_driver_manifest(result["stops"], store_geo).to_csv(index=False).encode("utf-8"),
                        file_name=f"driver_routes_{today.strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        type="primary",
                        key="driver_manifest_download",
                        help="Every truck's stop order with location and units to drop off - one sheet per dispatch.",
                    )

                    st.markdown("#### Route map")
                    st.caption("Scroll to zoom · drag to pan · hover a pin for location and drop-off quantity.")
                    render_route_map(
                        build_route_map(
                            result["routes"], store_geo, depot, depot_name, requests
                        )
                    )

                    with st.expander("View detailed manifest"):
                        st.dataframe(
                            requests.merge(
                                store_geo[["store", "location_name"]],
                                on="store",
                                how="left",
                            ),
                            hide_index=True,
                            width="stretch",
                        )

