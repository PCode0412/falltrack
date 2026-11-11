"""
train.py
Main training and subject-independent evaluation pipeline for SisFall.
Builds the feature pipeline, trains the HMM and baselines, runs explicit Viterbi decoding,
evaluates performance without data leakage, and persists all artifacts.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict

from src.data_loader import (
    scan_dataset_directory,
    get_subject_split,
    load_raw_recording,
    STATE_MAPPING
)
from src.preprocessing import preprocess_recording, segment_recording
from src.features import extract_window_features, FEATURE_NAMES
from src.hmm_model import ActivityGaussianHMM
from src.baseline_model import ThresholdFallDetector, BaselineRandomForest
from src.evaluation import evaluate_window_level, evaluate_event_level

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SisFall_dataset")


def process_recordings_list(recordings_df: pd.DataFrame, max_count: int = None) -> List[Dict]:
    """Loads, preprocesses, and segments recordings into sequences of windows."""
    if max_count and len(recordings_df) > max_count:
        recordings_df = recordings_df.sample(n=max_count, random_state=42)
        
    processed_sequences = []
    for _, row in recordings_df.iterrows():
        raw = load_raw_recording(row["filepath"])
        if raw is None:
            continue
        clean = preprocess_recording(raw)
        windows = segment_recording(clean)
        if not windows:
            continue
        
        # Raw features for this recording sequence
        X_seq_raw = np.array([extract_window_features(w) for w in windows], dtype=np.float32)
        y_seq = np.array([w["state_id"] for w in windows], dtype=np.int32)
        
        processed_sequences.append({
            "info": raw["info"],
            "activity_name": raw["activity_name"],
            "is_fall": raw["info"]["is_fall"],
            "duration": raw["duration"],
            "windows": windows,
            "X_raw": X_seq_raw,
            "y": y_seq,
            # Store downsampled sensor streams for visualization in the dashboard
            "vis_time": clean["time"][::5],  # 40 Hz downsample for fast dashboard rendering
            "vis_adxl": clean["adxl_filt"][::5],
            "vis_gyro": clean["gyro_filt"][::5],
            "vis_adxl_mag": clean["adxl_mag"][::5],
            "vis_gyro_mag": clean["gyro_mag"][::5],
        })
    return processed_sequences


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("SisFall HMM Fall Detection & HAR - Training and Evaluation Pipeline")
    print("=" * 70)
    
    # 1. Dataset Indexing & Subject Split
    print("\n[Step 1/6] Scanning SisFall dataset directory...")
    df = scan_dataset_directory(DATA_DIR)
    print(f"Total valid recording files found: {len(df)}")
    
    all_subjects = sorted(df["subject"].unique())
    train_subs, val_subs, test_subs = get_subject_split(all_subjects)
    print(f"Subject-Independent Split: {len(train_subs)} Train, {len(val_subs)} Val, {len(test_subs)} Test")
    print(f"  - Train Subjects: {train_subs}")
    print(f"  - Val Subjects:   {val_subs}")
    print(f"  - Test Subjects:  {test_subs}")
    
    train_df = df[df["subject"].isin(train_subs)]
    val_df = df[df["subject"].isin(val_subs)]
    test_df = df[df["subject"].isin(test_subs)]
    
    # 2. Balanced Training Selection & Feature Extraction
    print("\n[Step 2/6] Loading and segmenting training sequences...")
    # Balanced sample across activities: up to 15 trials per activity in training
    train_sampled = train_df.groupby("activity", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 18), random_state=42)
    )
    print(f"Selected {len(train_sampled)} balanced recordings for training (covering all 34 activities).")
    
    t0 = time.time()
    train_seqs = process_recordings_list(train_sampled)
    print(f"Processed {len(train_seqs)} training sequences in {time.time() - t0:.2f}s.")
    
    # Extract all training windows for scaler fitting
    X_train_raw = np.concatenate([s["X_raw"] for s in train_seqs], axis=0)
    y_train = np.concatenate([s["y"] for s in train_seqs], axis=0)
    print(f"Total training windows: {len(X_train_raw)}, Feature dimension: {X_train_raw.shape[1]}")
    print("Training State Distribution:")
    for s_id, s_name in STATE_MAPPING.items():
        cnt = int(np.sum(y_train == s_id))
        pct = (cnt / len(y_train)) * 100
        print(f"  - State {s_id} ({s_name}): {cnt} windows ({pct:.1f}%)")
        
    # Fit StandardScaler strictly on training data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    
    # Apply scaling to training sequences
    for s in train_seqs:
        s["X_scaled"] = scaler.transform(s["X_raw"])
        
    # 3. Model Training: Hidden Markov Model
    print("\n[Step 3/6] Fitting Gaussian Hidden Markov Model...")
    t0 = time.time()
    hmm = ActivityGaussianHMM(n_states=5, n_features=len(FEATURE_NAMES), random_state=42)
    seqs_X_train = [s["X_scaled"] for s in train_seqs]
    seqs_y_train = [s["y"] for s in train_seqs]
    hmm.fit(seqs_X_train, seqs_y_train)
    print(f"Gaussian HMM fitted in {time.time() - t0:.2f}s.")
    
    # 4. Baseline Models Training
    print("\n[Step 4/6] Training Baseline Classifiers on the same training set...")
    rf_baseline = BaselineRandomForest(n_estimators=100, max_depth=12, random_state=42)
    rf_baseline.fit(scaler.transform(X_train_raw), y_train)
    thresh_detector = ThresholdFallDetector(acc_threshold_g=2.8, gyro_threshold_dps=120.0)
    print("Random Forest baseline and Rule-based Threshold detector prepared.")
    
    # 5. Held-Out Subject Evaluation
    print("\n[Step 5/6] Evaluating on held-out test subjects (Strictly Unseen)...")
    # Evaluate across test subjects (up to 300 test recordings for comprehensive validation)
    test_sampled = test_df.groupby("activity", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 10), random_state=42)
    )
    print(f"Evaluating on {len(test_sampled)} held-out test recordings ({len(test_subs)} unseen subjects)...")
    
    t0 = time.time()
    test_seqs = process_recordings_list(test_sampled)
    print(f"Test sequences loaded and preprocessed in {time.time() - t0:.2f}s.")
    
    # Apply scaler to test sequences
    for s in test_seqs:
        s["X_scaled"] = scaler.transform(s["X_raw"])
        
    # Evaluate HMM with Explicit Viterbi Decoding
    all_y_true = []
    all_y_pred_hmm = []
    all_y_pred_rf = []
    
    hmm_recording_results = []
    rf_recording_results = []
    thresh_recording_results = []
    
    for s in test_seqs:
        X_scaled = s["X_scaled"]
        y_true = s["y"]
        
        # 1. HMM with explicit log-domain Viterbi decoding
        dec_states, dec_labels, log_prob, posteriors = hmm.viterbi_decode(X_scaled)
        s["decoded_states"] = dec_states
        s["decoded_labels"] = dec_labels
        s["posteriors"] = posteriors
        
        alert_info = hmm.predict_fall_alert(dec_states, posteriors)
        s["fall_alert"] = alert_info
        
        all_y_true.append(y_true)
        all_y_pred_hmm.append(dec_states)
        
        hmm_recording_results.append({
            "recording_info": s["info"],
            "activity_name": s["activity_name"],
            "is_fall": s["is_fall"],
            "duration": s["duration"],
            "fall_detected": alert_info["fall_detected"],
            "alert_window": alert_info["alert_window"],
            "delay_sec": alert_info["delay_sec"]
        })
        
        # 2. Random Forest Baseline
        rf_preds = rf_baseline.predict(X_scaled)
        all_y_pred_rf.append(rf_preds)
        rf_fall = rf_baseline.predict_fall_alert(X_scaled)
        rf_recording_results.append({
            "recording_info": s["info"],
            "activity_name": s["activity_name"],
            "is_fall": s["is_fall"],
            "duration": s["duration"],
            "fall_detected": rf_fall,
            "delay_sec": None
        })
        
        # 3. Threshold Baseline
        thresh_preds, thresh_fall = thresh_detector.predict_sequence(s["X_raw"], FEATURE_NAMES)
        thresh_recording_results.append({
            "recording_info": s["info"],
            "activity_name": s["activity_name"],
            "is_fall": s["is_fall"],
            "duration": s["duration"],
            "fall_detected": thresh_fall,
            "delay_sec": None
        })
        
    y_true_flat = np.concatenate(all_y_true)
    y_pred_hmm_flat = np.concatenate(all_y_pred_hmm)
    y_pred_rf_flat = np.concatenate(all_y_pred_rf)
    
    # Calculate Metrics
    hmm_window_metrics = evaluate_window_level(y_true_flat, y_pred_hmm_flat)
    rf_window_metrics = evaluate_window_level(y_true_flat, y_pred_rf_flat)
    
    hmm_event_metrics = evaluate_event_level(hmm_recording_results)
    rf_event_metrics = evaluate_event_level(rf_recording_results)
    thresh_event_metrics = evaluate_event_level(thresh_recording_results)
    
    print("\n" + "=" * 50)
    print("HELD-OUT TEST RESULTS (Subject-Independent):")
    print("=" * 50)
    print("\n--- Event-Level Fall Detection (Primary Deliverable) ---")
    print(f"HMM (Viterbi)   - Sensitivity: {hmm_event_metrics['event_metrics']['sensitivity']*100:.2f}%, Specificity: {hmm_event_metrics['event_metrics']['specificity']*100:.2f}%, Precision: {hmm_event_metrics['event_metrics']['precision']*100:.2f}%, F1: {hmm_event_metrics['event_metrics']['f1_score']*100:.2f}%")
    print(f"Random Forest   - Sensitivity: {rf_event_metrics['event_metrics']['sensitivity']*100:.2f}%, Specificity: {rf_event_metrics['event_metrics']['specificity']*100:.2f}%, Precision: {rf_event_metrics['event_metrics']['precision']*100:.2f}%, F1: {rf_event_metrics['event_metrics']['f1_score']*100:.2f}%")
    print(f"Threshold Rule  - Sensitivity: {thresh_event_metrics['event_metrics']['sensitivity']*100:.2f}%, Specificity: {thresh_event_metrics['event_metrics']['specificity']*100:.2f}%, Precision: {thresh_event_metrics['event_metrics']['precision']*100:.2f}%, F1: {thresh_event_metrics['event_metrics']['f1_score']*100:.2f}%")
    
    print("\n--- Window-Level Fall Detection ---")
    print(f"HMM (Viterbi)   - Sensitivity: {hmm_window_metrics['binary_metrics']['sensitivity']*100:.2f}%, Specificity: {hmm_window_metrics['binary_metrics']['specificity']*100:.2f}%, F1: {hmm_window_metrics['binary_metrics']['f1_score']*100:.2f}%")
    print(f"Random Forest   - Sensitivity: {rf_window_metrics['binary_metrics']['sensitivity']*100:.2f}%, Specificity: {rf_window_metrics['binary_metrics']['specificity']*100:.2f}%, F1: {rf_window_metrics['binary_metrics']['f1_score']*100:.2f}%")
    
    print(f"\nHMM Mean Detection Latency: {hmm_event_metrics['event_metrics']['mean_detection_delay_sec']:.2f} seconds")
    print(f"HMM False Alarms per Hour:  {hmm_event_metrics['event_metrics']['false_alarms_per_hour']:.2f} / hr")
    
    # 6. Saving Artifacts
    print("\n[Step 6/6] Persisting model and evaluation artifacts...")
    with open(os.path.join(MODELS_DIR, "hmm_model.json"), "w") as f:
        json.dump(hmm.to_dict(), f, indent=2)
        
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(rf_baseline.clf, os.path.join(MODELS_DIR, "baseline_rf.joblib"))
    
    eval_summary = {
        "dataset": "SisFall",
        "split_strategy": "Subject-Independent (Unseen Test Subjects)",
        "train_subjects": train_subs,
        "val_subjects": val_subs,
        "test_subjects": test_subs,
        "features": FEATURE_NAMES,
        "states": STATE_MAPPING,
        "hmm_window_metrics": hmm_window_metrics,
        "hmm_event_metrics": hmm_event_metrics,
        "rf_window_metrics": rf_window_metrics,
        "rf_event_metrics": rf_event_metrics,
        "thresh_event_metrics": thresh_event_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
        
    # Save a curated sample of test recordings for instant interactive simulation in the dashboard
    # Select 25 diverse recordings (covers walking, sitting, stumbles, and various falls)
    curated_sample = []
    for s in test_seqs:
        curated_sample.append({
            "info": s["info"],
            "activity_name": s["activity_name"],
            "is_fall": s["is_fall"],
            "duration": s["duration"],
            "vis_time": s["vis_time"],
            "vis_adxl": s["vis_adxl"],
            "vis_gyro": s["vis_gyro"],
            "vis_adxl_mag": s["vis_adxl_mag"],
            "vis_gyro_mag": s["vis_gyro_mag"],
            "X_raw": s["X_raw"],
            "X_scaled": s["X_scaled"],
            "y": s["y"],
            "decoded_states": s["decoded_states"],
            "decoded_labels": s["decoded_labels"],
            "posteriors": s["posteriors"],
            "fall_alert": s["fall_alert"]
        })
    joblib.dump(curated_sample[:50], os.path.join(MODELS_DIR, "test_recordings_sample.joblib"))
    
    print("\nAll models and evaluation artifacts successfully saved to:", MODELS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
