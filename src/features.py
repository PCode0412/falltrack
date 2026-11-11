"""
src/features.py
Extracts window-level statistical, kinematic, and energetic features from wearable inertial signals.
Manages feature ordering, scaling (leakage-safe), and persistence.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional

FEATURE_NAMES = [
    # Accelerometer Per-Axis Statistics
    "acc_x_mean", "acc_y_mean", "acc_z_mean",
    "acc_x_std",  "acc_y_std",  "acc_z_std",
    # Acceleration Magnitude & Impact Metrics
    "acc_mag_mean",
    "acc_mag_std",
    "acc_mag_max",       # Peak Impact Acceleration
    "acc_mag_min",       # Pre-impact Freefall / Weightlessness dip
    "acc_mag_ptp",       # Peak-to-peak amplitude
    "acc_rms",           # Root Mean Square of acceleration
    "acc_sma",           # Signal Magnitude Area
    "acc_energy",        # Energy of acceleration signal
    # Gyroscope Per-Axis Statistics
    "gyro_x_mean", "gyro_y_mean", "gyro_z_mean",
    "gyro_x_std",  "gyro_y_std",  "gyro_z_std",
    # Gyroscope Magnitude & Rotational Dynamics
    "gyro_mag_mean",
    "gyro_mag_std",
    "gyro_mag_max",      # Peak Rotational Velocity
    "gyro_rms"           # Gyroscope RMS
]


def extract_window_features(window: Dict) -> np.ndarray:
    """
    Extracts a 24-dimensional feature vector from a single 2-second sensor window.
    """
    adxl = window["adxl"]        # (N, 3) in g
    gyro = window["gyro"]        # (N, 3) in deg/s
    adxl_mag = window["adxl_mag"] # (N,) in g
    gyro_mag = window["gyro_mag"] # (N,) in deg/s
    
    # 1. Accelerometer per-axis mean and std
    acc_mean = np.mean(adxl, axis=0)
    acc_std = np.std(adxl, axis=0)
    
    # 2. Acceleration magnitude descriptors
    acc_mag_mean = float(np.mean(adxl_mag))
    acc_mag_std = float(np.std(adxl_mag))
    acc_mag_max = float(np.max(adxl_mag))
    acc_mag_min = float(np.min(adxl_mag))
    acc_mag_ptp = float(acc_mag_max - acc_mag_min)
    acc_rms = float(np.sqrt(np.mean(np.square(adxl_mag))))
    acc_sma = float(np.mean(np.sum(np.abs(adxl), axis=1)))
    acc_energy = float(np.mean(np.sum(np.square(adxl), axis=1)))
    
    # 3. Gyroscope per-axis mean and std
    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)
    
    # 4. Gyroscope magnitude descriptors
    gyro_mag_mean = float(np.mean(gyro_mag))
    gyro_mag_std = float(np.std(gyro_mag))
    gyro_mag_max = float(np.max(gyro_mag))
    gyro_rms = float(np.sqrt(np.mean(np.square(gyro_mag))))
    
    feat = [
        acc_mean[0], acc_mean[1], acc_mean[2],
        acc_std[0],  acc_std[1],  acc_std[2],
        acc_mag_mean, acc_mag_std, acc_mag_max, acc_mag_min, acc_mag_ptp,
        acc_rms, acc_sma, acc_energy,
        gyro_mean[0], gyro_mean[1], gyro_mean[2],
        gyro_std[0],  gyro_std[1],  gyro_std[2],
        gyro_mag_mean, gyro_mag_std, gyro_mag_max, gyro_rms
    ]
    
    return np.array(feat, dtype=np.float32)


def extract_dataset_features(
    windows: List[Dict], 
    scaler: Optional[StandardScaler] = None, 
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts features for an entire list of windows and applies standard scaling.
    Scaling parameters are strictly fitted on training data when fit_scaler=True.
    """
    X_list = []
    y_list = []
    for w in windows:
        f = extract_window_features(w)
        X_list.append(f)
        y_list.append(w["state_id"])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, y, scaler
