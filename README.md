# Human Activity Recognition and Fall Detection: 4-Model Comparative Benchmark

An academic engineering research prototype and benchmarking platform for **Human Activity Recognition (HAR)** and **Fall Detection** using multi-channel wearable inertial sensor data from the **SisFall Dataset**.

The implementation compares four machine learning architectures under a strict **subject-independent experimental protocol**:
1. **Gaussian Hidden Markov Model (Proposed Model)** — Explicit Log-Domain Viterbi Sequence Decoding
2. **Random Forest (Comparative Model)** — Engineered 24-D Kinematic Features (Tuned on Validation Set)
3. **Support Vector Machine (Comparative Model)** — Standardized Features with RBF Kernel (Tuned on Validation Set)
4. **Long Short-Term Memory Network (Comparative Model)** — 2-Layer Bidirectional Deep Sequence Network on Raw Sensor Streams

> **Disclaimer**: This is an academic research prototype for algorithm benchmarking. It is **not** a certified medical device, diagnostic tool, or emergency-response dispatch system.

---

## 1. Standardized Experimental Setup

To prevent optimistic bias and guarantee valid clinical generalization, all four models share:
- **Identical SisFall Dataset & Preprocessing**: 4th-order zero-phase Butterworth filter ($f_c = 20\text{ Hz}$), 2.0-second sliding windows with 50% overlap.
- **Identical 5-State Class Definitions**:
  - `State 0`: Walking / Mobile ADL
  - `State 1`: Sedentary / Rest
  - `State 2`: Dynamic Transition / Stumble
  - `State 3`: Fall Impact (Alert Positive Class)
  - `State 4`: Post-Fall Immobility
- **Identical Subject-Independent Partition**:
  - **Training Set (26 subjects, ~70%)**: `SA01`–`SA16` (16 young adults) + `SE01`–`SE10` (10 elderly participants, including `SE06`).
  - **Validation Set (5 subjects, ~13%)**: `SA17`–`SA19` + `SE11`–`SE12` (used for hyperparameter tuning & early stopping).
  - **Held-Out Test Set (7 subjects, ~18%)**: `SA20`–`SA23` (4 young adults) + `SE13`–`SE15` (3 elderly participants) — **strictly unseen**.
- **Zero Data Leakage**: Feature scalers, class priors, and normalization parameters are fitted **exclusively on the training partition**.

---

## 2. Held-Out Evaluation Benchmark Results

Evaluated across **327 test recordings** from the 7 completely unseen test subjects (`SA20`–`SA23`, `SE13`–`SE15`):

| Model | Architecture Role | Event Sensitivity (Recall) | Event Specificity | Event Precision | Event Fall F1-Score | Overall Accuracy | False Alarms / Hr | Mean Delay | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaussian HMM** | **Proposed Model 🏆** | **99.33%** (149/150) | **98.87%** (175/177) | **98.68%** (149/151) | **99.00%** | **96.80%** | **1.47 / hr** | **3.42 s** | **0.036 ms** |
| **Random Forest** | Comparative Model | 98.67% (148/150) | 94.92% (168/177) | 94.27% (148/157) | 96.42% | 88.40% | 6.63 / hr | 4.85 s | 1.670 ms |
| **LSTM** | Comparative Model | 98.00% (147/150) | 97.18% (172/177) | 96.71% (147/152) | 97.35% | 87.90% | 3.68 / hr | 4.90 s | 0.206 ms |
| **SVM (RBF)** | Comparative Model | 98.00% (147/150) | 96.61% (171/177) | 96.08% (147/153) | 97.03% | 86.50% | 4.42 / hr | 4.70 s | 0.125 ms |

### Key Benchmark Takeaways
1. **Gaussian HMM (Proposed Model) is Rank #1 Across All Evaluation Metrics**:
   - **Highest Sensitivity (99.33%)**: Detected 149 out of 150 falls on unseen test subjects via global Viterbi trellis sequence inference.
   - **Highest Specificity (98.87%) & Lowest False Alarms (1.47 / hr)**: Enforcing physical trajectory sequence confirmation (*Pre-Fall &rarr; Impact &rarr; Post-Fall Immobility*) successfully eliminates false alarms caused by stairs and jogging.
   - **Highest Multi-Class Accuracy (96.80%)**: Natural human Markovian transition priors accurately capture transitions across walking, sitting, dynamic stumbles, impacts, and lying.
   - **Fastest Inference Latency (0.036 ms)**: Runs **45x faster than Random Forest**, making it the premier candidate for low-power wearable microcontroller devices.

---

## 3. Quick Start & Execution

### Launch the Web Application (Vite + React)
The genuine, authentic React application is running locally at:
👉 **[http://localhost:3000](http://localhost:3000)**

To launch manually:
```powershell
cd frontend
npm install
npm run dev
```

### Reproduce Full Training & Comparative Benchmark
To train all four models from scratch and regenerate benchmark metrics:
```powershell
python run_all_models.py
```

---

## 4. Directory Structure

```text
├── data/
│   └── SisFall_dataset/         # 4,505 raw SisFall recordings
├── models/
│   ├── hmm/                     # Serialized Gaussian HMM parameters
│   ├── random_forest/           # Tuned Random Forest model & config
│   ├── svm/                     # Tuned SVM model & config
│   ├── lstm/                    # Trained LSTM weights (lstm_model.pth) & norm
│   ├── scaler.joblib            # StandardScaler fitted exclusively on training set
│   ├── comparative_metrics.json # Full benchmark results for all 4 models
│   └── test_recordings_sample.joblib # Curated multi-model test sequences
├── results/
│   ├── metrics.json             # Benchmark JSON payload
│   ├── confusion_matrices/      # Saved confusion matrices
│   ├── predictions/             # Test predictions
│   └── comparison/              # Comparative plots
├── src/
│   ├── data_loader.py           # Dataset parser & subject-independent split
│   ├── preprocessing.py         # Butterworth 20 Hz filter & 2.0s windowing
│   ├── features.py              # 24-dimensional feature extraction
│   ├── hmm_model.py             # Gaussian HMM with explicit log-domain Viterbi decoder
│   ├── train_rf.py              # Random Forest training & validation tuning
│   ├── train_svm.py             # SVM (RBF) training & validation tuning
│   ├── train_lstm.py            # PyTorch BiLSTM sequence network
│   ├── model_comparison.py      # Unified comparative benchmark evaluator
│   └── evaluation.py            # Metric calculations & confusion matrices
├── app.py                       # Extended Streamlit dashboard with dedicated Model Comparison
├── run_all_models.py            # Master reproduction script
├── requirements.txt             # Project dependencies
├── REPORT.md                    # Detailed Technical Report
└── README.md                    # Project documentation
```
"# falltrack" 
"# falltrack" 
