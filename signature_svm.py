import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import config

def train_model(signatures, labels):
    if len(signatures) < 10:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(signatures)
    # LinearSVC is much faster than SVC(kernel='linear')
    svm = LinearSVC(C=config.SVM_C, random_state=42, max_iter=1000)
    svm.fit(X_scaled, labels)
    return svm, scaler

def predict_decision(svm, scaler, signature):
    if svm is None:
        return 0.0
    X = signature.reshape(1, -1)
    X_scaled = scaler.transform(X)
    # LinearSVC has decision_function
    return svm.decision_function(X_scaled)[0]
