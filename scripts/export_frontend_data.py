"""
scripts/export_frontend_data.py
Exports dataset metadata, benchmark results, and curated test recording sensor streams
to frontend/public/data/ for high-speed, zero-dependency, authentic web rendering.
"""

import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import SUBJECT_METADATA, ACTIVITY_METADATA, STATE_MAPPING, STATE_COLORS
from src.features import FEATURE_NAMES

PUBLIC_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)
    print("Exporting data for Vite frontend...")
    
    # 1. Export Benchmark Metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(PUBLIC_DATA_DIR, "benchmark.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  -> Exported benchmark.json")
    
    # 2. Export Subjects
    subjects_list = []
    for s_id, s_data in SUBJECT_METADATA.items():
        is_test = s_id in ["SA20", "SA21", "SA22", "SA23", "SE13", "SE14", "SE15"]
        is_val = s_id in ["SA17", "SA18", "SA19", "SE11", "SE12"]
        subjects_list.append({
            "id": s_id,
            "group": s_data["group"],
            "age": s_data["age"],
            "height": s_data["height"],
            "weight": s_data["weight"],
            "gender": s_data["gender"],
            "split": "Held-Out Test" if is_test else ("Validation" if is_val else "Training")
        })
    with open(os.path.join(PUBLIC_DATA_DIR, "subjects.json"), "w") as f:
        json.dump(subjects_list, f, indent=2)
    print("  -> Exported subjects.json")
    
    # 3. Export Activities
    activities_list = []
    for a_id, a_data in ACTIVITY_METADATA.items():
        activities_list.append({
            "code": a_id,
            "name": a_data["name"],
            "type": a_data["type"],
            "category": a_data["category"],
            "duration_sec": a_data["duration_sec"]
        })
    with open(os.path.join(PUBLIC_DATA_DIR, "activities.json"), "w") as f:
        json.dump(activities_list, f, indent=2)
    print("  -> Exported activities.json")
    
    # 4. Export Curated Recordings
    curated = joblib.load(os.path.join(MODELS_DIR, "test_recordings_sample.joblib"))
    recordings_list = []
    
    for s in curated:
        rec_info = s["info"]
        time_arr = s["vis_time"].tolist()
        adxl = s["vis_adxl"].tolist()
        gyro = s["vis_gyro"].tolist()
        adxl_mag = s["vis_adxl_mag"].tolist()
        gyro_mag = s["vis_gyro_mag"].tolist()
        
        y_true = s["y"].tolist()
        dec_hmm = s["decoded_states_hmm"].tolist() if "decoded_states_hmm" in s else s["decoded_states"].tolist()
        preds_rf = s["preds_rf"].tolist() if "preds_rf" in s else dec_hmm
        preds_svm = s["preds_svm"].tolist() if "preds_svm" in s else dec_hmm
        preds_lstm = s["preds_lstm"].tolist() if "preds_lstm" in s else dec_hmm
        
        post_hmm = s["posteriors_hmm"].tolist() if "posteriors_hmm" in s else s["posteriors"].tolist()
        
        peak_acc = float(np.max(s["vis_adxl_mag"]))
        
        # Windows details
        n_w = len(y_true)
        windows_meta = []
        for w_i in range(n_w):
            windows_meta.append({
                "window": w_i,
                "start_time": round(w_i * 1.0, 1),
                "end_time": round(w_i * 1.0 + 2.0, 1),
                "y_true": int(y_true[w_i]),
                "pred_hmm": int(dec_hmm[w_i]),
                "pred_rf": int(preds_rf[w_i]),
                "pred_svm": int(preds_svm[w_i]),
                "pred_lstm": int(preds_lstm[w_i]),
                "hmm_posteriors": [round(float(p), 4) for p in post_hmm[w_i]],
                "peak_g": round(float(s["X_raw"][w_i, FEATURE_NAMES.index("acc_mag_max")]), 2)
            })
            
        recordings_list.append({
            "filename": rec_info["filename"],
            "activity": rec_info["activity"],
            "activity_name": s["activity_name"],
            "subject": rec_info["subject"],
            "trial": rec_info["trial"],
            "is_fall": bool(s["is_fall"]),
            "duration": float(s["duration"]),
            "peak_acc_g": round(peak_acc, 2),
            "time": [round(t, 2) for t in time_arr],
            "acc_x": [round(v[0], 3) for v in adxl],
            "acc_y": [round(v[1], 3) for v in adxl],
            "acc_z": [round(v[2], 3) for v in adxl],
            "acc_mag": [round(m, 3) for m in adxl_mag],
            "gyro_x": [round(v[0], 2) for v in gyro],
            "gyro_y": [round(v[1], 2) for v in gyro],
            "gyro_z": [round(v[2], 2) for v in gyro],
            "gyro_mag": [round(m, 2) for m in gyro_mag],
            "windows": windows_meta,
            "y_true": y_true,
            "preds_hmm": dec_hmm,
            "preds_rf": preds_rf,
            "preds_svm": preds_svm,
            "preds_lstm": preds_lstm,
            "fall_alert_hmm": s.get("fall_alert_hmm", s.get("fall_alert", {"fall_detected": False}))
        })
        
    with open(os.path.join(PUBLIC_DATA_DIR, "recordings.json"), "w") as f:
        json.dump(recordings_list, f)
    print(f"  -> Exported recordings.json ({len(recordings_list)} full test recordings)")
    
    print("Frontend data export complete!")

if __name__ == "__main__":
    main()
