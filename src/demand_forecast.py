"""
demand_forecast.py
===================
Short-horizon demand forecast overlaid on the Risk Pooling demand-volatility
view (Tab 2). Not a new layer in the A/B/C decomposition - just a forward-
looking annotation on Layer B's existing recent-demand sparklines.

Method: Holt-Winters exponential smoothing (statsmodels) with weekly
seasonality when there's enough history, falling back to a linear trend
extrapolation otherwise. Not Prophet/ARIMA: Holt-Winters fits in
milliseconds per store, cheap enough to run live when a SKU is selected,
so no offline precompute step is needed the way Layers A-C have one.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.logging_config import get_logger

logger = get_logger(__name__)

MIN_SEASONAL_OBS = 14  # >= 2 full weekly cycles before trusting a seasonal fit


def forecast_demand(series: pd.Series, horizon: int = 7) -> pd.Series:
    """Forecast `horizon` days beyond `series`' last date.

    `series` needs a DatetimeIndex at daily frequency. Negative forecasts are
    clipped to 0 (demand can't be negative). Returns an empty Series if there
    isn't enough history to forecast anything meaningful.
    """
    values = series.to_numpy(dtype=float)
    if len(values) < 2:
        return pd.Series(dtype=float)

    last_date = series.index.max()
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    if len(values) >= MIN_SEASONAL_OBS and values.std() > 0:
        try:
            model = ExponentialSmoothing(
                values, trend="add", seasonal="add", seasonal_periods=7, damped_trend=True,
            ).fit(optimized=True)
            forecast = np.asarray(model.forecast(horizon))
            return pd.Series(np.clip(forecast, 0, None), index=future_index)
        except Exception:
            logger.exception("Holt-Winters fit failed for a %d-day series, using linear trend", len(values))

    # Fallback: linear trend through whatever history is available.
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    forecast = intercept + slope * np.arange(len(values), len(values) + horizon)
    return pd.Series(np.clip(forecast, 0, None), index=future_index)
