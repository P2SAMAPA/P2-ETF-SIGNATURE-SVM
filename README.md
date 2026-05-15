# Signature Kernel SVM Engine

Classifies next‑day return sign using path signatures and an SVM with RBF kernel. For each ETF, we compute lead‑lag signatures (depth 3) over rolling windows (63, 126, 252 days). An SVM is trained on historical signature‑label pairs (positive/negative). The decision function value is the confidence score. Higher positive = stronger bullish signal.

- **Windows evaluated:** 63, 126, 252 days (best per ETF)
- **Model:** SVM (RBF kernel) on truncated signature features
- **Output:** top 3 ETFs per universe by decision value, with chosen window

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
