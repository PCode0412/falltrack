"""
src/data_loader.py
Loads recordings, labels, and subject metadata for the SisFall Dataset.
Handles raw parsing, physical unit scaling, and subject-independent splits.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Sampling frequency of SisFall sensors
SAMPLING_RATE_HZ = 200

# Physical unit scaling constants per SisFall specification
# ADXL345: +-16g, 13-bit -> (2 * 16) / 2^13 = 32 / 8192 = 0.00390625 g/bit
SCALE_ADXL345 = 32.0 / 8192.0

# ITG3200: +-2000 deg/s, 16-bit -> (2 * 2000) / 2^16 = 4000 / 65536 = 0.06103515625 deg/s/bit
SCALE_ITG3200 = 4000.0 / 65536.0

# MMA8451Q: +-8g, 14-bit -> (2 * 8) / 2^14 = 16 / 16384 = 0.0009765625 g/bit
SCALE_MMA8451Q = 16.0 / 16384.0

# Standard gravity conversion (g to m/s^2)
STANDARD_GRAVITY = 9.80665

# Subject metadata (extracted from SisFall Readme)
SUBJECT_METADATA = {
    # Young Adults (19-30 years)
    "SA01": {"age": 26, "height": 165, "weight": 53.0, "gender": "F", "group": "Adult"},
    "SA02": {"age": 23, "height": 176, "weight": 58.5, "gender": "M", "group": "Adult"},
    "SA03": {"age": 19, "height": 156, "weight": 48.0, "gender": "F", "group": "Adult"},
    "SA04": {"age": 23, "height": 170, "weight": 72.0, "gender": "M", "group": "Adult"},
    "SA05": {"age": 22, "height": 172, "weight": 69.5, "gender": "M", "group": "Adult"},
    "SA06": {"age": 21, "height": 169, "weight": 58.0, "gender": "M", "group": "Adult"},
    "SA07": {"age": 21, "height": 156, "weight": 63.0, "gender": "F", "group": "Adult"},
    "SA08": {"age": 21, "height": 149, "weight": 41.5, "gender": "F", "group": "Adult"},
    "SA09": {"age": 24, "height": 165, "weight": 64.0, "gender": "M", "group": "Adult"},
    "SA10": {"age": 21, "height": 177, "weight": 67.0, "gender": "M", "group": "Adult"},
    "SA11": {"age": 19, "height": 170, "weight": 80.5, "gender": "M", "group": "Adult"},
    "SA12": {"age": 25, "height": 153, "weight": 47.0, "gender": "F", "group": "Adult"},
    "SA13": {"age": 22, "height": 157, "weight": 55.0, "gender": "F", "group": "Adult"},
    "SA14": {"age": 27, "height": 160, "weight": 46.0, "gender": "F", "group": "Adult"},
    "SA15": {"age": 25, "height": 160, "weight": 52.0, "gender": "F", "group": "Adult"},
    "SA16": {"age": 20, "height": 169, "weight": 61.0, "gender": "F", "group": "Adult"},
    "SA17": {"age": 23, "height": 182, "weight": 75.0, "gender": "M", "group": "Adult"},
    "SA18": {"age": 23, "height": 181, "weight": 73.0, "gender": "M", "group": "Adult"},
    "SA19": {"age": 30, "height": 170, "weight": 76.0, "gender": "M", "group": "Adult"},
    "SA20": {"age": 30, "height": 150, "weight": 42.0, "gender": "F", "group": "Adult"},
    "SA21": {"age": 30, "height": 183, "weight": 68.0, "gender": "M", "group": "Adult"},
    "SA22": {"age": 19, "height": 158, "weight": 50.5, "gender": "F", "group": "Adult"},
    "SA23": {"age": 24, "height": 156, "weight": 48.0, "gender": "F", "group": "Adult"},
    # Elderly People (60-75 years)
    "SE01": {"age": 71, "height": 171, "weight": 102.0, "gender": "M", "group": "Elderly"},
    "SE02": {"age": 75, "height": 150, "weight": 57.0, "gender": "F", "group": "Elderly"},
    "SE03": {"age": 62, "height": 150, "weight": 51.0, "gender": "F", "group": "Elderly"},
    "SE04": {"age": 63, "height": 160, "weight": 59.0, "gender": "F", "group": "Elderly"},
    "SE05": {"age": 63, "height": 165, "weight": 72.0, "gender": "M", "group": "Elderly"},
    "SE06": {"age": 60, "height": 163, "weight": 79.0, "gender": "M", "group": "Elderly"},
    "SE07": {"age": 65, "height": 168, "weight": 76.0, "gender": "M", "group": "Elderly"},
    "SE08": {"age": 68, "height": 163, "weight": 72.0, "gender": "F", "group": "Elderly"},
    "SE09": {"age": 66, "height": 167, "weight": 65.0, "gender": "M", "group": "Elderly"},
    "SE10": {"age": 64, "height": 156, "weight": 66.0, "gender": "F", "group": "Elderly"},
    "SE11": {"age": 66, "height": 169, "weight": 63.0, "gender": "F", "group": "Elderly"},
    "SE12": {"age": 69, "height": 164, "weight": 56.5, "gender": "M", "group": "Elderly"},
    "SE13": {"age": 65, "height": 171, "weight": 72.5, "gender": "M", "group": "Elderly"},
    "SE14": {"age": 67, "height": 163, "weight": 58.0, "gender": "M", "group": "Elderly"},
    "SE15": {"age": 64, "height": 150, "weight": 50.0, "gender": "F", "group": "Elderly"},
}

# Complete Activity Descriptions
ACTIVITY_METADATA = {
    # Activities of Daily Living (ADLs)
    "D01": {"name": "Walking slowly", "type": "ADL", "category": "Walking", "duration_sec": 100},
    "D02": {"name": "Walking quickly", "type": "ADL", "category": "Walking", "duration_sec": 100},
    "D03": {"name": "Jogging slowly", "type": "ADL", "category": "Jogging", "duration_sec": 100},
    "D04": {"name": "Jogging quickly", "type": "ADL", "category": "Jogging", "duration_sec": 100},
    "D05": {"name": "Walking upstairs and downstairs slowly", "type": "ADL", "category": "Walking", "duration_sec": 25},
    "D06": {"name": "Walking upstairs and downstairs quickly", "type": "ADL", "category": "Walking", "duration_sec": 25},
    "D07": {"name": "Slowly sit in a half height chair and up", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D08": {"name": "Quickly sit in a half height chair and up", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D09": {"name": "Slowly sit in a low height chair and up", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D10": {"name": "Quickly sit in a low height chair and up", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D11": {"name": "Sitting, trying to get up, collapse into chair", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D12": {"name": "Sitting, lying slowly, and sit again", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D13": {"name": "Sitting, lying quickly, and sit again", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D14": {"name": "On back roll to lateral and back", "type": "ADL", "category": "Sedentary", "duration_sec": 12},
    "D15": {"name": "Standing, slowly bending at knees and getting up", "type": "ADL", "category": "Dynamic", "duration_sec": 12},
    "D16": {"name": "Standing, bending without knees and getting up", "type": "ADL", "category": "Dynamic", "duration_sec": 12},
    "D17": {"name": "Get into car, remain seated, get out", "type": "ADL", "category": "Dynamic", "duration_sec": 25},
    "D18": {"name": "Stumble while walking", "type": "ADL", "category": "Dynamic", "duration_sec": 12},
    "D19": {"name": "Gently jump without falling", "type": "ADL", "category": "Dynamic", "duration_sec": 12},
    # Falls
    "F01": {"name": "Fall forward while walking (slip)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F02": {"name": "Fall backward while walking (slip)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F03": {"name": "Lateral fall while walking (slip)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F04": {"name": "Fall forward while walking (trip)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F05": {"name": "Fall forward while jogging (trip)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F06": {"name": "Vertical fall while walking (fainting)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F07": {"name": "Fall while walking (table dampen, fainting)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F08": {"name": "Fall forward when trying to get up", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F09": {"name": "Lateral fall when trying to get up", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F10": {"name": "Fall forward when trying to sit down", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F11": {"name": "Fall backward when trying to sit down", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F12": {"name": "Lateral fall when trying to sit down", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F13": {"name": "Fall forward while sitting (fainting/sleep)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F14": {"name": "Fall backward while sitting (fainting/sleep)", "type": "Fall", "category": "Fall", "duration_sec": 15},
    "F15": {"name": "Lateral fall while sitting (fainting/sleep)", "type": "Fall", "category": "Fall", "duration_sec": 15},
}

# HMM Activity State Definitions
STATE_MAPPING = {
    0: "Walking / Mobile ADL",
    1: "Sedentary / Rest",
    2: "Dynamic Transition / Stumble",
    3: "Fall Impact",
    4: "Post-Fall Immobility"
}

STATE_COLORS = {
    0: "#2563EB",  # Muted Blue
    1: "#64748B",  # Slate
    2: "#D97706",  # Amber
    3: "#DC2626",  # Crimson (Alert)
    4: "#7C3AED"   # Purple
}


def parse_filename(filepath: str) -> Optional[Dict[str, str]]:
    """
    Parses a SisFall filename into activity code, subject ID, and trial ID.
    Example: 'D01_SA01_R01.txt' -> {'activity': 'D01', 'subject': 'SA01', 'trial': 'R01'}
    """
    basename = os.path.basename(filepath)
    pattern = r"([DF]\d{2})_([A-Z]{2}\d{2})_(R\d{2})\.txt"
    match = re.match(pattern, basename, re.IGNORECASE)
    if not match:
        return None
    act, sub, trial = match.groups()
    return {
        "filename": basename,
        "filepath": filepath,
        "activity": act.upper(),
        "subject": sub.upper(),
        "trial": trial.upper(),
        "is_fall": act.upper().startswith("F")
    }


def load_raw_recording(filepath: str) -> Optional[Dict]:
    """
    Reads a raw SisFall recording file and converts raw sensor bits to physical units.
    Returns:
        dict containing:
            - time: 1D array of time stamps in seconds
            - adxl: (N, 3) acceleration array in g [X, Y, Z]
            - itg: (N, 3) angular velocity array in deg/s [X, Y, Z]
            - mma: (N, 3) secondary acceleration array in g [X, Y, Z]
            - info: metadata dictionary
    """
    info = parse_filename(filepath)
    if not info:
        return None
    
    try:
        # SisFall files contain comma-separated integers, lines may end with semicolon or whitespace
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = [l.strip().rstrip(';').split(',') for l in f if l.strip()]
        
        if not lines or len(lines) < 20:
            return None
        
        raw_data = np.array([[float(x) for x in l] for l in lines], dtype=np.float32)
        
        if raw_data.shape[1] != 9:
            # Malformed row dimension
            return None
        
        # Sensor channels based on dataset documentation:
        # Columns 0, 1, 2: ADXL345 (accel 1, waist)
        # Columns 3, 4, 5: ITG3200 (gyro, waist)
        # Columns 6, 7, 8: MMA8451Q (accel 2, waist)
        adxl_g = raw_data[:, 0:3] * SCALE_ADXL345
        itg_dps = raw_data[:, 3:6] * SCALE_ITG3200
        mma_g = raw_data[:, 6:9] * SCALE_MMA8451Q
        
        n_samples = len(raw_data)
        time_sec = np.arange(n_samples) / SAMPLING_RATE_HZ
        
        act_info = ACTIVITY_METADATA.get(info["activity"], {
            "name": info["activity"], "type": "Fall" if info["is_fall"] else "ADL", "category": "Unknown"
        })
        sub_info = SUBJECT_METADATA.get(info["subject"], {
            "age": 0, "height": 0, "weight": 0, "gender": "Unknown", "group": "Unknown"
        })
        
        return {
            "info": info,
            "activity_name": act_info["name"],
            "activity_category": act_info["category"],
            "subject_meta": sub_info,
            "time": time_sec,
            "adxl": adxl_g,
            "gyro": itg_dps,
            "mma": mma_g,
            "n_samples": n_samples,
            "duration": n_samples / SAMPLING_RATE_HZ
        }
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def get_subject_split(all_subjects: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Defines a strict subject-independent split (Train / Val / Test).
    Ensures subjects in the test set never appear in training or validation.
    """
    # Young adults (SA01 - SA23)
    sa_all = sorted([s for s in all_subjects if s.startswith("SA")])
    # Elderly people (SE01 - SE15)
    se_all = sorted([s for s in all_subjects if s.startswith("SE")])
    
    # Train: 16 Young + 10 Elderly = 26 subjects (~70%)
    # Note: SE06 (the judo expert who also performed falls) is in Train to enrich elderly fall representation
    train_subs = [s for s in sa_all[:16]] + [s for s in se_all[:10]]
    
    # Val: 3 Young + 2 Elderly = 5 subjects (~13%)
    val_subs = [s for s in sa_all[16:19]] + [s for s in se_all[10:12]]
    
    # Test: 4 Young + 3 Elderly = 7 subjects (~18%)
    test_subs = [s for s in sa_all[19:23]] + [s for s in se_all[12:15]]
    
    return train_subs, val_subs, test_subs


def scan_dataset_directory(data_dir: str) -> pd.DataFrame:
    """
    Scans the SisFall directory, parses all valid recording paths and creates an index dataframe.
    """
    pattern = os.path.join(data_dir, "**", "*.txt")
    files = glob.glob(pattern, recursive=True)
    
    records = []
    for fp in files:
        parsed = parse_filename(fp)
        if parsed:
            act_info = ACTIVITY_METADATA.get(parsed["activity"], {})
            sub_info = SUBJECT_METADATA.get(parsed["subject"], {})
            records.append({
                "filepath": fp,
                "filename": parsed["filename"],
                "activity": parsed["activity"],
                "subject": parsed["subject"],
                "trial": parsed["trial"],
                "is_fall": parsed["is_fall"],
                "activity_name": act_info.get("name", parsed["activity"]),
                "category": act_info.get("category", "Unknown"),
                "age": sub_info.get("age", None),
                "gender": sub_info.get("gender", None),
                "group": sub_info.get("group", None)
            })
    
    df = pd.DataFrame(records)
    return df
