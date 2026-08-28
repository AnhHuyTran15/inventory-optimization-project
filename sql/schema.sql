-- ============================================================================
-- schema.sql
-- ============================================================================
-- Postgres schema for the inventory optimization pipeline.
--
-- NOT required for the notebooks/app to run today (everything currently
-- reads/writes CSV under data/processed/). This schema exists for the
-- planned "Add Database" upgrade step: swap pd.read_csv/to_csv calls in
-- src/*.py for SQLAlchemy sessions against these tables, without changing
-- the shape of the data itself.
--
-- Run with:  psql -U <user> -d <dbname> -f sql/schema.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS demand_daily (
    id              BIGSERIAL PRIMARY KEY,
    store           SMALLINT NOT NULL,
    item            SMALLINT NOT NULL,
    sale_date       DATE NOT NULL,
    sales           INTEGER NOT NULL CHECK (sales >= 0),
    UNIQUE (store, item, sale_date)
);
CREATE INDEX IF NOT EXISTS idx_demand_daily_store_item ON demand_daily (store, item);

COMMENT ON TABLE demand_daily IS 'Raw daily sales, loaded from train.csv (Kaggle Store-Item Demand).';


CREATE TABLE IF NOT EXISTS shipment_tracking (
    booking_id                  TEXT PRIMARY KEY,
    gps_provider                TEXT,
    shipment_type                TEXT,
    booking_date                 TIMESTAMP NOT NULL,
    vehicle_registration         TEXT,
    origin_location               TEXT,
    origin_latitude               DOUBLE PRECISION,
    origin_longitude              DOUBLE PRECISION,
    destination_location          TEXT,
    destination_latitude          DOUBLE PRECISION,
    destination_longitude         DOUBLE PRECISION,
    trip_start_date               TIMESTAMP,
    trip_end_date                 TIMESTAMP,
    vehicle_type                  TEXT,
    transportation_distance_km    DOUBLE PRECISION,
    customer_name                 TEXT,
    supplier_name                 TEXT,
    material_shipped               TEXT,
    lead_time_days                 DOUBLE PRECISION GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (trip_end_date - booking_date)) / 86400.0
    ) STORED
);

COMMENT ON TABLE shipment_tracking IS 'Raw GPS shipment tracking, loaded from the logistics tracking xlsx.';


CREATE TABLE IF NOT EXISTS model_inputs (
    store                       SMALLINT NOT NULL,
    item                        SMALLINT NOT NULL,
    demand_mean                 DOUBLE PRECISION NOT NULL,
    demand_std                  DOUBLE PRECISION NOT NULL,
    demand_min                  INTEGER,
    demand_max                  INTEGER,
    n_days                      INTEGER,
    dispersion_index            DOUBLE PRECISION,
    suggested_distribution      TEXT,
    assigned_vehicle_type       TEXT,
    lead_time_mean_days         DOUBLE PRECISION,
    lead_time_std_days          DOUBLE PRECISION,
    holding_cost_rate_annual    DOUBLE PRECISION,
    order_cost_fixed            DOUBLE PRECISION,
    target_service_level        DOUBLE PRECISION,
    unit_cost_placeholder       DOUBLE PRECISION,
    built_at                    TIMESTAMP DEFAULT now(),
    PRIMARY KEY (store, item)
);

COMMENT ON TABLE model_inputs IS 'Output of src/data_prep.py - joined demand + sampled lead time + cost assumptions.';


CREATE TABLE IF NOT EXISTS layer_a_policies (
    store          SMALLINT NOT NULL,
    item           SMALLINT NOT NULL,
    s_reorder      INTEGER NOT NULL,
    s_order_up_to  INTEGER NOT NULL,
    x_max_used     INTEGER,
    iterations     INTEGER,
    final_diff     DOUBLE PRECISION,
    converged      BOOLEAN,
    violations     INTEGER,
    n_states       INTEGER,
    solved_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (store, item),
    FOREIGN KEY (store, item) REFERENCES model_inputs (store, item)
);

COMMENT ON TABLE layer_a_policies IS 'Output of src/layer_a_dp.py - one (s,S) policy per store-item pair.';


CREATE TABLE IF NOT EXISTS layer_b_pooling_results (
    item                        SMALLINT PRIMARY KEY,
    n_stores                    SMALLINT,
    ss_decentralized            DOUBLE PRECISION,
    ss_pooled                   DOUBLE PRECISION,
    pct_savings                 DOUBLE PRECISION,
    mean_pairwise_correlation   DOUBLE PRECISION,
    computed_at                 TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE layer_b_pooling_results IS 'Output of src/layer_b_pooling.py - decentralized vs pooled safety stock per item.';


CREATE TABLE IF NOT EXISTS delivery_requests (
    id              BIGSERIAL PRIMARY KEY,
    store           SMALLINT NOT NULL,
    item            SMALLINT NOT NULL,
    request_date    DATE NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    fulfilled       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE delivery_requests IS 'Layer C input - generated from Layer A/B policy vs. simulated on-hand inventory.';


CREATE TABLE IF NOT EXISTS routes (
    id                  BIGSERIAL PRIMARY KEY,
    route_date          DATE NOT NULL,
    vehicle_id          TEXT NOT NULL,
    stop_sequence       INTEGER NOT NULL,
    store               SMALLINT NOT NULL,
    distance_from_prev_km  DOUBLE PRECISION,
    delivered_quantity  INTEGER,
    created_at          TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE routes IS 'Layer C output - one row per stop per route, from the VRP solve.';


CREATE TABLE IF NOT EXISTS run_events (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE run_events IS 'Structured run log, mirrors logs/runs.jsonl - useful once queries beat grepping text logs.';
CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_events (event_type);
