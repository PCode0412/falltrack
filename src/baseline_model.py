"""
src/baseline_model.py
Baseline Fall Detection Models:
1. Rule-Based Threshold Detector (Impact Peak & Inactivity check)
2. Conventional Window-Level Random Forest Classifier
Evaluated on the exact same held-out test data for direct comparison with HMM + Viterbi.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional


class ThresholdFallDetector:
    """
    Standard biomedical rule-based baseline:
    Triggers a fall if acceleration magnitude exceeds an impact threshold (default 2.8g)
    and angular velocity exceeds 150 deg/s.
    """
    def __init__(self, acc_threshold_g: float = 2.8, gyro_threshold_dps: float = 120.0):
        self.acc_threshold_g = acc_threshold_g
        self.gyro_threshold_dps = gyro_threshold_dps

    def predict_window(self, features: np.ndarray, feature_names: List[str]) -> int:
        """Returns 1 if fall is detected in window, 0 otherwise."""
        acc_max_idx = feature_names.index("acc_mag_max")
        gyro_max_idx = feature_names.index("gyro_mag_max")
        
        acc_peak = features[acc_max_idx]
        gyro_peak = features[gyro_max_idx]
        
        if acc_peak >= self.acc_threshold_g and gyro_peak >= self.gyro_threshold_dps:
            return 1
        return 0

    def predict_sequence(self, X_raw: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, bool]:
        """Predicts window sequence and recording-level fall alert."""
        preds = np.array([self.predict_window(row, feature_names) for row in X_raw], dtype=int)
        fall_detected = bool(np.any(preds == 1))
        return preds, fall_detected


class BaselineRandomForest:
    """
    Supervised Machine Learning Window Classifier (Random Forest).
    Classifies each window independently without sequence/temporal context.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_fall_alert(self, X: np.ndarray, fall_state_id: int = 3) -> bool:
        preds = self.predict(X)
        return bool(np.any(preds == fall_state_id))
