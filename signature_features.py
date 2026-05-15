import numpy as np
import pandas as pd

def lead_lag_path(prices_series):
    if len(prices_series) < 2:
        return None
    lead = prices_series[1:].values
    lag = prices_series[:-1].values
    return np.column_stack([lead, lag])

def signature_depth2(path):
    """
    Compute signature of a 2D path up to depth 2.
    Returns 6 features: (1st order: 2, 2nd order: 4)
    """
    if path is None or len(path) < 2:
        return None
    inc = np.diff(path, axis=0)
    # 1st order
    sig1 = np.sum(inc, axis=0)          # (2,)
    # 2nd order (four double integrals)
    L = inc.shape[0]
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
    return np.concatenate([sig1, sig2])   # shape (6,)

def rolling_signatures(price_series, window):
    """
    For a single ETF price series (pandas Series), compute signature for each day
    using the last `window` days. Returns a list of signature vectors and labels.
    """
    signatures = []
    labels = []
    for i in range(window, len(price_series)-1):
        # window data
        window_prices = price_series.iloc[i-window:i]
        path = lead_lag_path(window_prices)
        if path is None:
            continue
        sig = signature_depth2(path)
        if sig is None:
            continue
        signatures.append(sig)
        # label: next day return sign
        next_ret = price_series.iloc[i+1] - price_series.iloc[i]
        labels.append(1 if next_ret > 0 else 0)
    return np.array(signatures), np.array(labels)
