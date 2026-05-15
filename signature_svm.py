import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import config

def train_model(signatures, labels):
    if len(signatures) < 10:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(signatures)
    # SVC with RBF kernel (non‑linear)
    svm = SVC(kernel=config.SVM_KERNEL, C=config.SVM_C, gamma=config.SVM_GAMMA,
              probability=False, random_state=42)
    svm.fit(X_scaled, labels)
    return svm, scaler

def predict_decision(svm, scaler, signature):
    if svm is None:
        return 0.0
    X = signature.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return svm.decision_function(X_scaled)[0]
