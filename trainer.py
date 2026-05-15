import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from signature_features import compute_signature_for_etf, lead_lag_path, compute_signature_manual
from signature_svm import train_svm_for_etf, predict_decision

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Signature Kernel SVM) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        if prices.empty or len(prices) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}

        for win in config.WINDOWS:
            if len(prices) < win + 50:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")

            for etf in tickers:
                if etf not in prices.columns:
                    continue
                signatures = []
                labels = []
                # Build rolling dataset for this ETF
                for i in range(win, len(prices)-1):
                    window_prices = prices[etf].iloc[i-win:i]
                    sig = compute_signature_for_etf(prices, etf, win, depth=config.SIG_DEPTH)
                    if sig is None:
                        continue
                    signatures.append(sig)
                    next_ret = prices[etf].iloc[i+1] - prices[etf].iloc[i]
                    label = 1 if next_ret > 0 else 0
                    labels.append(label)
                if len(signatures) < 20:
                    continue
                svm, scaler = train_svm_for_etf(signatures, labels)
                if svm is None:
                    continue
                # Predict for most recent window
                last_sig = compute_signature_for_etf(prices, etf, win, depth=config.SIG_DEPTH)
                if last_sig is None:
                    continue
                decision = predict_decision(svm, scaler, last_sig)
                if etf not in best_per_etf or decision > best_per_etf[etf][0]:
                    best_per_etf[etf] = (decision, win)

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (decision, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "decision": float(decision), "best_window": win})
            full_scores[ticker] = {"decision": float(decision), "best_window": win}
        print(f"  Top 3 ETFs by SVM decision value: {[e['ticker'] for e in top_etfs]}")
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
    print("\n=== Signature Kernel SVM Engine complete ===")

if __name__ == "__main__":
    main()
