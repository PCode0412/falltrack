"""
src/evaluation.py
Calculates window-level and event-level evaluation metrics:
Sensitivity, Specificity, Precision, Fall F1-Score, Confusion Matrices,
False Alarm Rates, Detection Delays, and Error Analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix, classification_report


def calculate_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Computes sensitivity, specificity, precision, and F1-score with safe zero-division handling.
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "accuracy": float(accuracy)
    }


def evaluate_window_level(y_true: np.ndarray, y_pred: np.ndarray, fall_state_id: int = 3) -> Dict:
    """
    Evaluates fall detection at the individual window level (binary: Fall State vs Other).
    """
    bin_true = (y_true == fall_state_id).astype(int)
    bin_pred = (y_pred == fall_state_id).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).ravel()
    metrics = calculate_binary_metrics(tp, fp, fn, tn)
    
    # 5-class confusion matrix and classification report
    classes = [0, 1, 2, 3, 4]
    mc_cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    return {
        "binary_metrics": metrics,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "multiclass_cm": mc_cm.tolist()
    }


def evaluate_event_level(
    recordings_results: List[Dict], 
    total_adl_hours: Optional[float] = None
) -> Dict:
    """
    Evaluates fall detection at the recording/event level.
    A fall recording is a True Positive if an alert was triggered.
    An ADL recording is a False Positive if an alert was falsely triggered.
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    delays = []
    error_analysis = {
        "false_positives": [],
        "false_negatives": [],
        "true_positives": [],
        "true_negatives": []
    }
    
    total_adl_duration_sec = 0.0
    
    for res in recordings_results:
        is_fall_ground_truth = res["is_fall"]
        alert_triggered = res["fall_detected"]
        rec_info = res["recording_info"]
        duration = res.get("duration", 15.0)
        
        if not is_fall_ground_truth:
            total_adl_duration_sec += duration
            
        if is_fall_ground_truth and alert_triggered:
            tp += 1
            if res.get("delay_sec") is not None:
                delays.append(res["delay_sec"])
            error_analysis["true_positives"].append({
                "filename": rec_info["filename"],
                "activity": rec_info["activity"],
                "subject": rec_info["subject"],
                "delay_sec": res.get("delay_sec")
            })
        elif is_fall_ground_truth and not alert_triggered:
            fn += 1
            error_analysis["false_negatives"].append({
                "filename": rec_info["filename"],
                "activity": rec_info["activity"],
                "subject": rec_info["subject"],
                "description": res.get("activity_name", rec_info["activity"])
            })
        elif not is_fall_ground_truth and alert_triggered:
            fp += 1
            error_analysis["false_positives"].append({
                "filename": rec_info["filename"],
                "activity": rec_info["activity"],
                "subject": rec_info["subject"],
                "description": res.get("activity_name", rec_info["activity"]),
                "alert_window": res.get("alert_window")
            })
        else:
            tn += 1
            error_analysis["true_negatives"].append({
                "filename": rec_info["filename"],
                "activity": rec_info["activity"],
                "subject": rec_info["subject"]
            })
            
    metrics = calculate_binary_metrics(tp, fp, fn, tn)
    
    # False alarms per hour of ADL monitoring
    total_hours = (total_adl_duration_sec / 3600.0) if total_adl_duration_sec > 0 else (total_adl_hours or 1.0)
    metrics["false_alarms_per_hour"] = float(fp / total_hours) if total_hours > 0 else 0.0
    metrics["mean_detection_delay_sec"] = float(np.mean(delays)) if len(delays) > 0 else 0.0
    metrics["total_recordings"] = len(recordings_results)
    metrics["total_adl_hours"] = float(total_hours)
    
    return {
        "event_metrics": metrics,
        "error_analysis": error_analysis
    }
