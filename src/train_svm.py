"""
src/train_svm.py
Support Vector Machine (SVM) comparative model training and validation tuning.
Uses standardized window-level statistical features with RBF kernel and class weighting.
"""

import os
import json
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from typing import Dict, Tuple, List

from src.features import FEATURE_NAMES


def tune_and_train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = "models/svm"
) -> Tuple[SVC, Dict]:
    """
    Tunes Support Vector Machine (RBF kernel) hyperparameters on validation data,
    trains the optimal model, and saves the model artifact and configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Tuning SVM (RBF Kernel) Hyperparameters on Validation Set ---")
    
    # Subsample training data if very large for fast, responsive SVM quadratic optimization
    if len(X_train) > 6000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_train), size=6000, replace=False)
        X_tr_sub = X_train[sample_idx]
        y_tr_sub = y_train[sample_idx]
    else:
        X_tr_sub = X_train
        y_tr_sub = y_train
        
    param_grid = [
        {"C": 0.5, "gamma": "scale", "class_weight": "balanced"},
        {"C": 1.0, "gamma": "scale", "class_weight": "balanced"},
        {"C": 2.0, "gamma": 0.02, "class_weight": "balanced"},
    ]
    
    best_score = -1.0
    best_params = param_grid[1]
    
    for params in param_grid:
        clf = SVC(
            C=params["C"],
            kernel="rbf",
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            probability=True,
            random_state=42,
            max_iter=3000
        )
        clf.fit(X_tr_sub, y_tr_sub)
        preds_val = clf.predict(X_val)
        
        # Validation Fall class F1-score
        bin_true = (y_val == 3).astype(int)
        bin_pred = (preds_val == 3).astype(int)
        tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
        fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))
        fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))
        val_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        print(f"  Params: {params} -> Val Fall F1: {val_f1 * 100:.2f}%")
        if val_f1 > best_score:
            best_score = val_f1
            best_params = params
            
    print(f"Optimal SVM Parameters: {best_params} (Val Fall F1: {best_score*100:.2f}%)")
    
    # Train final SVM model
    best_svm = SVC(
        C=best_params["C"],
        kernel="rbf",
        gamma=best_params["gamma"],
        class_weight=best_params["class_weight"],
        probability=True,
        random_state=42,
        max_iter=4000
    )
    t0 = time.time()
    best_svm.fit(X_tr_sub, y_tr_sub)
    fit_time = time.time() - t0
    
    # Save model and configuration
    model_path = os.path.join(output_dir, "svm_model.joblib")
    joblib.dump(best_svm, model_path)
    
    config = {
        "model_name": "Support Vector Machine (SVM)",
        "type": "Comparative Margin-Based Classifier",
        "kernel": "rbf",
        "best_params": best_params,
        "validation_fall_f1": float(best_score),
        "fit_time_seconds": float(fit_time),
        "n_features": len(FEATURE_NAMES)
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    return best_svm, config
