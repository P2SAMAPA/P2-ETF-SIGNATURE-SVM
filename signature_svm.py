import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_svm_for_etf(signature_series, targets):
    """
    signature_series: list of signature vectors (each a numpy array)
    targets: list of binary labels (1 for positive next-day return, 0 for negative)
    Returns trained SVM model and scaler.
    """
    if len(signature_series) < 10:
        return None, None
    X = np.vstack(signature_series)
    y = np.array(targets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=False, random_state=42)
    svm.fit(X_scaled, y)
    return svm, scaler

def predict_decision(svm, scaler, signature):
    """Return decision function value (distance to hyperplane) for a new signature."""
    if svm is None:
        return 0.0
    X = signature.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return svm.decision_function(X_scaled)[0]
