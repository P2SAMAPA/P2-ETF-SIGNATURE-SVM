import numpy as np
import pandas as pd
import iisignature

def lead_lag_transform(prices, etf_name):
    """
    Create lead‑lag path for a single ETF.
    Returns a 2D array (2, window) where first row is lead, second is lag.
    """
    # For single asset, lead-lag path is: (t, price_t) and (t, price_{t-1})
    # Actually standard lead-lag: X = (X_t, X_{t-1}) for each t.
    # We'll generate a stream of 2D points: (price_t, price_{t-1})
    price_series = prices[etf_name].dropna().values
    if len(price_series) < 2:
        return None
    lead = price_series[1:]
    lag = price_series[:-1]
    path = np.column_stack([lead, lag])   # shape (window, 2)
    return path

def compute_signature(prices_df, etf_name, window, depth=3):
    """
    Compute truncated signature for a single ETF over the last `window` days.
    Returns a flat signature vector (size = sum_{k=1}^{depth} 2^k) for 2D path.
    """
    if len(prices_df) < window:
        return None
    # Take last window days of prices
    window_prices = prices_df.iloc[-window:][[etf_name]].copy()
    # Need at least 2 points for signature
    if len(window_prices) < 2:
        return None
    path = lead_lag_transform(window_prices, etf_name)
    if path is None:
        return None
    # iisignature expects a list of lists: [[x1,y1], [x2,y2], ...]
    sig = iisignature.sig(path, depth)
    return np.array(sig)

def compute_signatures_universe(prices_df, etf_names, window, depth=3):
    """
    Compute signatures for all ETFs in the universe for a given window.
    Returns dict {etf: signature vector}.
    """
    result = {}
    for etf in etf_names:
        sig = compute_signature(prices_df, etf, window, depth)
        if sig is not None:
            result[etf] = sig
    return result
