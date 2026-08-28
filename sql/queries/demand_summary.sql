-- demand_summary.sql
-- Sanity-check query: per store, total demand and item count, useful after
-- loading demand_daily to confirm row counts match the source CSV (913,000
-- rows total, 1,826 days per store-item pair, no gaps).

SELECT
    store,
    COUNT(DISTINCT item)               AS n_items,
    COUNT(*)                           AS n_rows,
    MIN(sale_date)                     AS first_date,
    MAX(sale_date)                     AS last_date,
    SUM(sales)                         AS total_units_sold,
    ROUND(AVG(sales)::numeric, 1)      AS avg_daily_sales_per_row
FROM demand_daily
GROUP BY store
ORDER BY store;
