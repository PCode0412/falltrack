"""
run_all_models.py
Master pipeline to train, tune, and evaluate all 4 models under the standardized setup:
1. Gaussian HMM (Proposed Model)
2. Random Forest (Comparative Model)
3. Support Vector Machine - RBF (Comparative Model)
4. Long Short-Term Memory - LSTM (Comparative Model)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch

from src.data_loader import (
    scan_dataset_directory,
    get_subject_split,
    load_raw_recording,
    STATE_MAPPING
)
from src.preprocessing import preprocess_recording, segment_recording
from src.features import extract_window_features, FEATURE_NAMES
from src.hmm_model import ActivityGaussianHMM
from src.train_rf import tune_and_train_rf
from src.train_svm import tune_and_train_svm
from src.train_lstm import train_lstm_network, prepare_raw_sensor_windows, ActivityLSTM
from src.model_comparison import run_comparative_benchmark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "SisFall_dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def process_split_recordings(recordings_df: pd.DataFrame, max_per_activity: int = 15) -> list:
    """Processes recordings into windowed sequences."""
    sampled_df = recordings_df.groupby("activity", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max_per_activity), random_state=42)
    )
    processed = []
    for _, row in sampled_df.iterrows():
        raw = load_raw_recording(row["filepath"])
        if raw is None:
            continue
        clean = preprocess_recording(raw)
        windows = segment_recording(clean)
        if not windows:
            continue
        X_raw = np.array([extract_window_features(w) for w in windows], dtype=np.float32)
        y = np.array([w["state_id"] for w in windows], dtype=np.int32)
        processed.append({
            "info": raw["info"],
            "activity_name": raw["activity_name"],
            "is_fall": raw["info"]["is_fall"],
            "duration": raw["duration"],
            "windows": windows,
            "X_raw": X_raw,
            "y": y,
            "vis_time": clean["time"][::5],
            "vis_adxl": clean["adxl_filt"][::5],
            "vis_gyro": clean["gyro_filt"][::5],
            "vis_adxl_mag": clean["adxl_mag"][::5],
            "vis_gyro_mag": clean["gyro_mag"][::5],
        })
    return processed


def main():
    print("=" * 70)
    print("STANDARDIZED 4-MODEL HAR & FALL DETECTION TRAINING PIPELINE")
    print("Models: 1. Gaussian HMM (Proposed) | 2. RF | 3. SVM | 4. LSTM")
    print("=" * 70)
    
    # 1. Dataset Indexing & Subject Split
    print("\n[Step 1/6] Scanning SisFall dataset and partitioning subjects...")
    df = scan_dataset_directory(DATA_DIR)
    all_subjects = sorted(df["subject"].unique())
    train_subs, val_subs, test_subs = get_subject_split(all_subjects)
    
    train_df = df[df["subject"].isin(train_subs)]
    val_df = df[df["subject"].isin(val_subs)]
    test_df = df[df["subject"].isin(test_subs)]
    
    print(f"Train Subjects ({len(train_subs)}): {train_subs[:6]} ...")
    print(f"Val Subjects   ({len(val_subs)}): {val_subs}")
    print(f"Test Subjects  ({len(test_subs)}): {test_subs}")
    
    # 2. Loading sequences
    print("\n[Step 2/6] Loading Train, Val, and Test sequences...")
    train_seqs = process_split_recordings(train_df, max_per_activity=15)
    val_seqs = process_split_recordings(val_df, max_per_activity=6)
    test_seqs = process_split_recordings(test_df, max_per_activity=10)
    
    print(f"Train Sequences: {len(train_seqs)}, Val Sequences: {len(val_seqs)}, Test Sequences: {len(test_seqs)}")
    
    # Flatten feature matrices
    X_train_raw = np.concatenate([s["X_raw"] for s in train_seqs], axis=0)
    y_train = np.concatenate([s["y"] for s in train_seqs], axis=0)
    
    X_val_raw = np.concatenate([s["X_raw"] for s in val_seqs], axis=0)
    y_val = np.concatenate([s["y"] for s in val_seqs], axis=0)
    
    # Fit StandardScaler strictly on training set
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    for s in train_seqs:
        s["X_scaled"] = scaler.transform(s["X_raw"])
    for s in val_seqs:
        s["X_scaled"] = scaler.transform(s["X_raw"])
    for s in test_seqs:
        s["X_scaled"] = scaler.transform(s["X_raw"])
        
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # Flatten window lists for LSTM training
    train_windows_all = [w for s in train_seqs for w in s["windows"]]
    val_windows_all = [w for s in val_seqs for w in s["windows"]]
    
    # 3. Model 1: Gaussian HMM (PROPOSED MODEL)
    print("\n[Step 3/6] Fitting Gaussian HMM (PROPOSED MODEL) with Viterbi Decoding...")
    hmm = ActivityGaussianHMM(n_states=5, n_features=len(FEATURE_NAMES), random_state=42)
    hmm.fit([s["X_scaled"] for s in train_seqs], [s["y"] for s in train_seqs])
    os.makedirs(os.path.join(MODELS_DIR, "hmm"), exist_ok=True)
    with open(os.path.join(MODELS_DIR, "hmm", "hmm_model.json"), "w") as f:
        json.dump(hmm.to_dict(), f, indent=2)
    # Also save in root models/
    with open(os.path.join(MODELS_DIR, "hmm_model.json"), "w") as f:
        json.dump(hmm.to_dict(), f, indent=2)
        
    # 4. Model 2: Random Forest (Comparative Model)
    print("\n[Step 4/6] Tuning & Training Random Forest (Comparative Model)...")
    rf_model, rf_config = tune_and_train_rf(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        output_dir=os.path.join(MODELS_DIR, "random_forest")
    )
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "baseline_rf.joblib"))
    
    # 5. Model 3: Support Vector Machine (Comparative Model)
    print("\n[Step 5/6] Tuning & Training SVM - RBF (Comparative Model)...")
    svm_model, svm_config = tune_and_train_svm(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        output_dir=os.path.join(MODELS_DIR, "svm")
    )
    
    # 6. Model 4: LSTM (Comparative Model)
    print("\n[Step 6/6] Training Bidirectional LSTM Deep Sequence Network (Comparative Model)...")
    lstm_model, lstm_config, lstm_norm = train_lstm_network(
        train_windows_all,
        val_windows_all,
        output_dir=os.path.join(MODELS_DIR, "lstm"),
        epochs=12,
        batch_size=64
    )
    
    # 7. Unified Comparative Benchmark across all 4 models
    comparison_results = run_comparative_benchmark(
        hmm_model=hmm,
        rf_model=rf_model,
        svm_model=svm_model,
        lstm_model=lstm_model,
        lstm_norm=lstm_norm,
        scaler=scaler,
        test_sequences=test_seqs
    )
    
    # 8. Curate multi-model test sequences for interactive dashboard demonstration
    print("\nGenerating curated multi-model test sequences for dashboard...")
    ch_mean = np.array(lstm_norm["mean"], dtype=np.float32)
    ch_std = np.array(lstm_norm["std"], dtype=np.float32)
    device = torch.device("cpu")
    lstm_model.eval()
    
    curated_sample = []
    with torch.no_grad():
        for s in test_seqs:
            # Predictions from all 4 models
            dec_states_hmm, dec_labels_hmm, _, post_hmm = hmm.viterbi_decode(s["X_scaled"])
            preds_rf = rf_model.predict(s["X_scaled"])
            preds_svm = svm_model.predict(s["X_scaled"])
            
            X_raw_seq, _ = prepare_raw_sensor_windows(s["windows"], target_seq_len=50)
            X_norm_seq = (X_raw_seq - ch_mean) / ch_std
            logits = lstm_model(torch.tensor(X_norm_seq, dtype=torch.float32).to(device))
            preds_lstm = torch.argmax(logits, dim=1).cpu().numpy()
            
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
                # Model predictions
                "decoded_states_hmm": dec_states_hmm,
                "decoded_labels_hmm": dec_labels_hmm,
                "posteriors_hmm": post_hmm,
                "fall_alert_hmm": hmm.predict_fall_alert(dec_states_hmm, post_hmm),
                
                "preds_rf": preds_rf,
                "preds_svm": preds_svm,
                "preds_lstm": preds_lstm,
                
                # Backwards compatible default keys pointing to proposed HMM
                "decoded_states": dec_states_hmm,
                "decoded_labels": dec_labels_hmm,
                "posteriors": post_hmm,
                "fall_alert": hmm.predict_fall_alert(dec_states_hmm, post_hmm)
            })
            
    joblib.dump(curated_sample[:50], os.path.join(MODELS_DIR, "test_recordings_sample.joblib"))
    print("Master pipeline finished successfully! All 4 models trained and evaluated.")


if __name__ == "__main__":
    main()
