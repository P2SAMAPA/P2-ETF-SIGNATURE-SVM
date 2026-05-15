import numpy as np
import pandas as pd

def lead_lag_path(prices_series):
    """
    Convert a price series to a lead‑lag path:
    Points: (price_t, price_{t-1}) for t = 1..N-1.
    Returns numpy array of shape (N-1, 2).
    """
    if len(prices_series) < 2:
        return None
    lead = prices_series[1:].values
    lag = prices_series[:-1].values
    return np.column_stack([lead, lag])

def compute_signature_manual(path, depth=3):
    """
    Compute truncated signature of a 2D path (list of points) up to depth 3.
    Uses iterated integrals via simple loop (for small depth).
    Returns a flat numpy array of length sum_{k=1}^{depth} 2^k.
    """
    if path is None or len(path) < 2:
        return None
    # Compute increments
    inc = np.diff(path, axis=0)  # shape (L, 2)
    L = inc.shape[0]
    # Precompute all iterated integrals recursively
    # For depth 1: integrals of dX and dY (just the total increment)
    sig1 = np.sum(inc, axis=0)  # (2,)
    # For depth 2: integrals of dX dX, dX dY, dY dX, dY dY
    # Using double loop
    sig2 = np.zeros(4)
    idx = 0
    for i in range(2):
        for j in range(2):
            total = 0.0
            for s in range(L):
                # Integral of dX_i from 0..s times dX_j at s
                prefix = np.sum(inc[:s+1, i]) if s >= 0 else 0.0
                total += prefix * inc[s, j]
            sig2[idx] = total
            idx += 1
    # For depth 3: triple integrals (2^3 = 8)
    sig3 = np.zeros(8)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for s2 in range(L):
                    # double integral up to s2
                    double = 0.0
                    for s1 in range(s2+1):
                        prefix = np.sum(inc[:s1+1, i]) if s1 >= 0 else 0.0
                        double += prefix * inc[s1, j]
                    total += double * inc[s2, k]
                sig3[idx] = total
                idx += 1
    return np.concatenate([sig1, sig2, sig3])

def compute_signature_for_etf(prices_df, etf, window, depth=3):
    """
    Compute signature for a single ETF over the last `window` days.
    """
    if len(prices_df) < window:
        return None
    window_prices = prices_df[etf].iloc[-window:]
    path = lead_lag_path(window_prices)
    if path is None:
        return None
    return compute_signature_manual(path, depth)
