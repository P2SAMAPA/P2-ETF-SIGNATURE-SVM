import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from signature_features import rolling_signatures
from signature_svm import train_model, predict_decision

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Signature SVM) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        if prices.empty or len(prices) < config.WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        for etf in tickers:
            if etf not in prices.columns:
                continue
            # Get price series
            price_series = prices[etf].dropna()
            if len(price_series) < config.WINDOW + 20:
                continue
            signatures, labels = rolling_signatures(price_series, config.WINDOW)
            if len(signatures) < 20:
                continue
            svm, scaler = train_model(signatures, labels)
            if svm is None:
                continue
            # Last signature (most recent window)
            last_window = price_series.iloc[-config.WINDOW:]
            path = lead_lag_path(last_window)
            if path is None:
                continue
            last_sig = signature_depth2(path)
            if last_sig is None:
                continue
            decision = predict_decision(svm, scaler, last_sig)
            best_per_etf[etf] = decision

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, decision in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "decision": float(decision)})
            full_scores[ticker] = float(decision)
        print(f"  Top 3 ETFs by decision value: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/sig_svm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Signature Kernel SVM Engine (fast) complete ===")

if __name__ == "__main__":
    main()
