-- lead_time_stats.sql
-- Recomputes the Vehicle-Type lead time table (the same logic as
-- src/data_prep.py::build_lead_time_stats, but in SQL) directly against
-- shipment_tracking, useful for a quick check without leaving the DB.
--
-- Filters out the same noise as the Python version:
--   - lead_time_days <= 1/24 (sub-1-hour GPS ping artifacts)
--   - top 1% outliers (computed via percentile_cont)

WITH filtered AS (
    SELECT vehicle_type, lead_time_days
    FROM shipment_tracking
    WHERE lead_time_days > (1.0 / 24)
),
cap AS (
    SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY lead_time_days) AS p99
    FROM filtered
)
SELECT
    f.vehicle_type,
    COUNT(*)                                    AS n_obs,
    ROUND(AVG(f.lead_time_days)::numeric, 2)     AS lead_time_mean_days,
    ROUND(STDDEV(f.lead_time_days)::numeric, 2)  AS lead_time_std_days
FROM filtered f, cap
WHERE f.lead_time_days <= cap.p99
GROUP BY f.vehicle_type
HAVING COUNT(*) >= 5
ORDER BY n_obs DESC;
