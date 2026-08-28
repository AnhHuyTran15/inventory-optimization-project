# Multi-Echelon Stochastic Inventory & Delivery Optimization

## SCQA

**Situation.**
A retail network operates 10 stores carrying 50 SKUs each (500 store-item pairs), replenished from a shared distribution network. Daily demand history (2013-2017, ~913K rows) and real shipment GPS-tracking history (lead times, vehicle types, origin/destination coordinates) are both available, but have never been used together to drive a replenishment decision.

**Complication.**
Standard heuristics — a fixed EOQ formula, a static reorder point — treat demand and lead time as known constants. In reality both are random: daily demand varies by up to ±55% around its mean (CV ≈ 0.55 across SKUs), and observed delivery lead time ranges from under a day to nearly 20 days depending on vehicle type. A closed-form EOQ answer cannot represent this uncertainty, cannot exploit the fact that store-level demand is imperfectly correlated (risk pooling), and says nothing about the physical constraint that replenishment requires a truck with finite capacity following a route.

**Question.**
Given real, measured demand variability and real, measured lead-time variability, what replenishment policy — how much safety stock, what reorder point, held where in the network, delivered by which routes — minimizes total system cost (holding + stockout + fixed ordering + transportation) while meeting a target service level?

**Answer (what this project builds).**
A three-layer decision model, each layer solved with a distinct OR technique, chained together and exposed through an interactive app:

| Layer | Technique | Question it answers |
|---|---|---|
| A — Single-echelon | Stochastic Dynamic Programming (Bellman equation, infinite-horizon value iteration) | What is the optimal (s, S) policy for *each* store-item pair, given its own measured demand distribution? |
| B — Multi-echelon | Risk-pooling / base-stock analysis (Eppen, 1979) | How much safety stock is saved by pooling demand across correlated stores instead of each store holding its own buffer? |
| C — Inventory-Routing | Decomposition: (s,S) policy → delivery requests → Capacitated VRP | Given the deliveries Layer A/B says are needed, what is the lowest-cost set of truck routes that satisfies them under vehicle capacity constraints? |

The baseline (EOQ, static reorder point) is kept in the results layer explicitly as a comparison point — the project's central claim is quantitative: *X% lower total cost than EOQ, at the same service level*, not just "we built a fancier model."

---

## Why this exists

This is a personal learning project, not a job-search portfolio piece — so it is deliberately over-scoped in places (a Postgres schema, a logging setup, a deploy pipeline) to intentionally practice skills adjacent to the OR modeling itself: data engineering discipline, reproducibility, and shipping something a non-technical person could open in a browser.

## Data sources

- `data/raw/train.csv` — Kaggle "Store Item Demand Forecasting Challenge" (5 years daily sales, 10 stores × 50 items, no gaps).
- `data/raw/Transportation__Logistics_Tracking_Dataset.xlsx` — GPS shipment tracking (3,585 bookings, origin/destination coordinates, vehicle type, booking/trip timestamps).

**Important caveat, stated up front rather than discovered later:** these two datasets do not share a key (no common SKU or store ID). Lead time is therefore drawn from the logistics dataset's empirical Vehicle-Type distribution and *assigned* to each store-item pair by weighted random sampling (seed = 42, reproducible) rather than being a real 1:1 measurement per SKU. This is documented explicitly in `notebooks/01_data_preparation.ipynb` and should be repeated in any write-up of results — it is a modeling assumption, not a hidden data leak.

## Repository structure

```
inventory-optimization-project/
├── README.md                       <- this file (SCQA + overview)
├── LICENSE                         <- MIT
├── requirements.txt
├── .gitignore
├── notebooks/                      <- one notebook per pipeline stage, run in order
│   ├── 00_scqa_overview.ipynb
│   ├── 01_data_preparation.ipynb
│   ├── 02_layer_a_dp_inventory.ipynb
│   ├── 03_layer_b_risk_pooling.ipynb
│   ├── 04_layer_c_irp_routing.ipynb
│   └── 05_results_comparison.ipynb
├── src/                            <- reusable modules imported by notebooks + app
│   ├── logging_config.py
│   ├── data_prep.py
│   ├── layer_a_dp.py
│   ├── layer_b_pooling.py
│   └── layer_c_routing.py
├── sql/
│   ├── schema.sql                  <- Postgres schema for processed outputs
│   └── queries/                    <- example analytical queries
├── app/
│   └── app.py                      <- Streamlit app (Layer A/B results, deployable as-is)
├── data/
│   ├── raw/                        <- original files (gitignored, see .gitignore)
│   └── processed/                  <- model_inputs.csv and other derived tables
├── logs/                           <- run logs (gitignored, kept via .gitkeep)
└── tests/
    └── test_layer_a.py             <- verifies the (s,S) structure holds (0 violations)
```

## How to run (VS Code)

1. `python -m venv .venv && source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Place the two raw data files into `data/raw/`
4. Open `notebooks/` in VS Code's Jupyter extension, run `00` → `05` in order
5. `streamlit run app/app.py` to launch the interactive app locally
6. To deploy: push to a public GitHub repo → connect at share.streamlit.io → done

## License

MIT — see `LICENSE`.
