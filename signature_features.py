import numpy as np
import pandas as pd

def lead_lag_path(prices_series):
    if len(prices_series) < 2:
        return None
    lead = prices_series[1:].values
    lag = prices_series[:-1].values
    return np.column_stack([lead, lag])

def signature_depth3(path):
    """
    Compute truncated signature of a 2D path up to depth 3.
    Returns 14 features: (2 + 4 + 8)
    """
    if path is None or len(path) < 2:
        return None
    inc = np.diff(path, axis=0)          # shape (L, 2)
    L = inc.shape[0]
    # Depth 1
    sig1 = np.sum(inc, axis=0)           # (2,)
    # Depth 2
    sig2 = np.zeros(4)
    idx = 0
    for i in range(2):
        for j in range(2):
            total = 0.0
            for s in range(L):
                prefix = np.sum(inc[:s+1, i])
                total += prefix * inc[s, j]
            sig2[idx] = total
            idx += 1
    # Depth 3
    sig3 = np.zeros(8)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for s2 in range(L):
                    double = 0.0
                    for s1 in range(s2+1):
                        prefix = np.sum(inc[:s1+1, i])
                        double += prefix * inc[s1, j]
                    total += double * inc[s2, k]
                sig3[idx] = total
                idx += 1
    return np.concatenate([sig1, sig2, sig3])

def rolling_signatures(price_series, window, depth=3):
    """
    For a single ETF price series, compute signatures for each rolling window.
    Returns (signatures, labels) where signatures is (n_samples, n_features).
    """
    signatures = []
    labels = []
    for i in range(window, len(price_series)-1):
        window_prices = price_series.iloc[i-window:i]
        path = lead_lag_path(window_prices)
        if path is None:
            continue
        if depth == 3:
            sig = signature_depth3(path)
        else:
            # fallback to depth 2
            from signature_features import signature_depth2
            sig = signature_depth2(path)
        if sig is None:
            continue
        signatures.append(sig)
        next_ret = price_series.iloc[i+1] - price_series.iloc[i]
        labels.append(1 if next_ret > 0 else 0)
    return np.array(signatures), np.array(labels)
