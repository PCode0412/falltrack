"""
src/train_rf.py
Random Forest comparative model training and validation tuning.
Uses engineered window-level statistical features and standard subject split.
"""

import os
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List

from src.features import FEATURE_NAMES
from src.evaluation import calculate_binary_metrics, evaluate_window_level, evaluate_event_level


def tune_and_train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = "models/random_forest"
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Tunes Random Forest hyperparameters on validation data, trains the optimal model,
    and saves the model artifact and configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Tuning Random Forest Hyperparameters on Validation Set ---")
    
    param_grid = [
        {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced"},
        {"n_estimators": 150, "max_depth": 16, "class_weight": "balanced"},
    ]
    
    best_score = -1.0
    best_params = param_grid[1]
    
    for params in param_grid:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight=params["class_weight"],
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        preds_val = clf.predict(X_val)
        
        # Validation score based on Fall class (State 3) F1-score
        bin_true = (y_val == 3).astype(int)
        bin_pred = (preds_val == 3).astype(int)
        tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
        fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))
        fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))
        tn = int(np.sum((bin_true == 0) & (bin_pred == 0)))
        val_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        print(f"  Params: {params} -> Val Fall F1: {val_f1 * 100:.2f}%")
        if val_f1 > best_score:
            best_score = val_f1
            best_params = params
            
    print(f"Optimal Random Forest Parameters: {best_params} (Val Fall F1: {best_score*100:.2f}%)")
    
    # Train final model with best parameters
    best_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        class_weight=best_params["class_weight"],
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    best_rf.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # Save model and configuration
    model_path = os.path.join(output_dir, "rf_model.joblib")
    joblib.dump(best_rf, model_path)
    
    config = {
        "model_name": "Random Forest",
        "type": "Comparative Baseline",
        "best_params": best_params,
        "validation_fall_f1": float(best_score),
        "fit_time_seconds": float(fit_time),
        "n_features": len(FEATURE_NAMES)
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    return best_rf, config
