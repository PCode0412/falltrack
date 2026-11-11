"""
src/preprocessing.py
Signal conditioning, zero-phase Butterworth filtering, segmentation into overlapping windows,
and documented window-state assignment rules.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Dict, Optional

# Preprocessing parameters
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF_HZ = 20.0  # Preserves human impact transients up to 20 Hz while removing jitter
WINDOW_DURATION_SEC = 2.0     # 2-second window
WINDOW_OVERLAP_RATIO = 0.5    # 50% overlap
STEP_DURATION_SEC = 1.0       # 1-second step

def apply_butterworth_filter(
    signal: np.ndarray, 
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ, 
    fs: float = 200.0, 
    order: int = BUTTERWORTH_ORDER
) -> np.ndarray:
    """
    Applies a zero-phase 4th order low-pass Butterworth filter.
    Zero-phase forward-backward filtering (filtfilt) ensures no phase delay or time shifting.
    """
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Filter along time axis (axis 0)
    if signal.ndim == 1:
        if len(signal) <= 3 * order:
            return signal
        return filtfilt(b, a, signal)
    else:
        if signal.shape[0] <= 3 * order:
            return signal
        filtered = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            filtered[:, ch] = filtfilt(b, a, signal[:, ch])
        return filtered


def preprocess_recording(rec: Dict) -> Dict:
    """
    Cleans and applies low-pass Butterworth filter to accelerometer and gyroscope signals.
    """
    rec_clean = dict(rec)
    rec_clean["adxl_filt"] = apply_butterworth_filter(rec["adxl"])
    rec_clean["gyro_filt"] = apply_butterworth_filter(rec["gyro"])
    rec_clean["mma_filt"] = apply_butterworth_filter(rec["mma"])
    
    # Pre-calculate magnitude vectors
    rec_clean["adxl_mag"] = np.linalg.norm(rec_clean["adxl_filt"], axis=1)
    rec_clean["gyro_mag"] = np.linalg.norm(rec_clean["gyro_filt"], axis=1)
    return rec_clean


def segment_recording(
    rec: Dict, 
    window_sec: float = WINDOW_DURATION_SEC, 
    step_sec: float = STEP_DURATION_SEC,
    fs: float = 200.0
) -> List[Dict]:
    """
    Segments continuous recording into overlapping windows and applies majority-state labeling.
    
    State mapping:
      0: Walking / Mobile ADL
      1: Sedentary / Rest
      2: Dynamic Transition / Stumble
      3: Fall Impact
      4: Post-Fall Immobility
    """
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    n_samples = rec["n_samples"]
    
    if n_samples < window_samples:
        return []
    
    adxl = rec["adxl_filt"]
    gyro = rec["gyro_filt"]
    adxl_mag = rec["adxl_mag"]
    gyro_mag = rec["gyro_mag"]
    time_arr = rec["time"]
    
    is_fall_file = rec["info"]["is_fall"]
    act_code = rec["info"]["activity"]
    act_cat = rec["activity_category"]
    
    # Determine impact peak timing if fall trial
    impact_sample = None
    if is_fall_file:
        # Find peak acceleration magnitude
        impact_sample = int(np.argmax(adxl_mag))
        impact_time = time_arr[impact_sample]
    
    windows = []
    start_idx = 0
    w_idx = 0
    
    while start_idx + window_samples <= n_samples:
        end_idx = start_idx + window_samples
        t_start = time_arr[start_idx]
        t_end = time_arr[end_idx - 1]
        
        # Determine window state label
        if not is_fall_file:
            # ADL recording
            if act_cat == "Walking" or act_code in ["D01", "D02", "D03", "D04", "D05", "D06"]:
                state_id = 0  # Walking / Mobile
            elif act_cat == "Sedentary" or act_code in ["D07", "D08", "D09", "D10", "D11", "D12", "D13", "D14"]:
                state_id = 1  # Sedentary / Rest
            else:
                state_id = 2  # Dynamic transition / Stumble / Jumps (D15-D19)
        else:
            # Fall recording sequence: Pre-fall ADL -> Fall Impact -> Post-fall Immobility
            # Check overlap with impact peak
            # A window is considered Fall Impact if the impact sample falls within the window
            if start_idx <= impact_sample < end_idx:
                state_id = 3  # Fall Impact
            elif end_idx <= impact_sample:
                # Pre-fall movement (typically walking or sitting depending on fall type)
                if act_code in ["F01", "F02", "F03", "F04", "F05", "F06", "F07"]:
                    state_id = 0  # Walking pre-fall
                elif act_code in ["F13", "F14", "F15"]:
                    state_id = 1  # Sitting pre-fall
                else:
                    state_id = 2  # Dynamic transition pre-fall (trying to stand/sit)
            else:
                # Post-fall phase: lying on the ground / immobility
                state_id = 4  # Post-Fall Immobility
        
        window_dict = {
            "window_index": w_idx,
            "start_sample": start_idx,
            "end_sample": end_idx,
            "start_time": t_start,
            "end_time": t_end,
            "adxl": adxl[start_idx:end_idx],
            "gyro": gyro[start_idx:end_idx],
            "adxl_mag": adxl_mag[start_idx:end_idx],
            "gyro_mag": gyro_mag[start_idx:end_idx],
            "state_id": state_id,
            "is_fall_window": (state_id == 3),
            "recording_info": rec["info"]
        }
        windows.append(window_dict)
        
        start_idx += step_samples
        w_idx += 1
        
    return windows
