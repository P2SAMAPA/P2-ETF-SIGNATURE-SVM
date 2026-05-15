import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from signature_features import compute_signatures_universe
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

        # For each ETF, we will compute signatures for all days (sliding window)
        # Instead of per‑ETF, we need a time‑series of signatures per ETF.
        # We'll loop over days: for each day t (starting from min_window), we compute signature for each ETF using the last `win` days.
        # Then we train an SVM for each window individually? Actually we want to train an SVM per ETF per window.
        # But that's many models. Simpler: For each window, we train ONE SVM that takes signatures from all ETFs as input? That doesn't work.
        # The correct approach: For each ETF, we build a dataset of its own signatures and next-day returns over time. Then train an SVM for that ETF. Then predict its next-day decision value.
        # We'll implement that: for each ETF, for a given window, we create a sequence of signatures (each from a rolling sub‑window) and corresponding labels (next day sign). Then train SVM. Then predict for the most recent signature.
        # This is computationally heavy but doable daily. We'll implement for each window.

        best_per_etf = {}  # ticker -> (best_decision, best_window)

        for win in config.WINDOWS:
            if len(prices) < win + 50:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")

            # For each ETF, build training data
            for etf in tickers:
                if etf not in prices.columns:
                    continue
                # Collect signatures and labels over time
                signatures = []
                labels = []
                # We need a rolling window: for each day i (from win to len(prices)-1), take prices[i-win:i] as window, compute signature for that ETF, label = sign(return on day i+1)
                for i in range(win, len(prices)-1):
                    window_prices = prices.iloc[i-win:i][[etf]].copy()
                    if len(window_prices) < 2:
                        continue
                    # Compute signature using lead‑lag transform
                    from signature_features import lead_lag_transform, compute_signature
                    # We'll compute directly using a helper
                    path = lead_lag_transform(window_prices, etf)
                    if path is None:
                        continue
                    sig = iisignature.sig(path, config.SIG_DEPTH)   # need import here
                    signatures.append(sig)
                    # Next day return sign
                    next_ret = prices[etf].iloc[i+1] - prices[etf].iloc[i]
                    label = 1 if next_ret > 0 else 0
                    labels.append(label)
                if len(signatures) < 20:
                    continue
                # Train SVM
                svm, scaler = train_svm_for_etf(signatures, labels)
                if svm is None:
                    continue
                # Predict for the most recent signature (last window)
                last_window = prices.iloc[-win:][[etf]].copy()
                last_sig = compute_signature(prices, etf, win, depth=config.SIG_DEPTH)
                if last_sig is None:
                    continue
                decision = predict_decision(svm, scaler, last_sig)
                # Store best per ETF
                if etf not in best_per_etf or decision > best_per_etf[etf][0]:
                    best_per_etf[etf] = (decision, win)

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort by decision value descending
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (decision, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({
                "ticker": ticker,
                "decision": float(decision),
                "best_window": win
            })
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
