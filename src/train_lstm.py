"""
src/train_lstm.py
Long Short-Term Memory (LSTM) deep sequence network for activity recognition and fall detection.
Preserves raw temporal ordering of accelerometer and gyroscope channels.
Implements dropout, early stopping, and validation-based model selection.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class SensorSequenceDataset(Dataset):
    """PyTorch Dataset for raw sensor sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ActivityLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier for 6-channel wearable sensor time series.
    Channels: [adxl_x, adxl_y, adxl_z, gyro_x, gyro_y, gyro_z]
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3):
        super(ActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Bidirectional produces 2 * hidden_dim
        lstm_out_dim = hidden_dim * 2
        # Concatenate global temporal average pooling and max pooling
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, 2 * hidden_dim)
        # Global mean and max pooling over the temporal dimension
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.fc(pooled)
        return logits


def prepare_raw_sensor_windows(windows: List[Dict], target_seq_len: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts raw 6-channel time series from window dictionaries downsampled to target_seq_len (e.g. 50 steps = 25 Hz).
    Channels: [adxl_x, adxl_y, adxl_z, gyro_x, gyro_y, gyro_z]
    """
    X_list = []
    y_list = []
    for w in windows:
        adxl = w["adxl"]  # (400, 3)
        gyro = w["gyro"]  # (400, 3)
        raw_6ch = np.concatenate([adxl, gyro], axis=1)  # (400, 6)
        
        # Subsample to target_seq_len
        step = max(1, len(raw_6ch) // target_seq_len)
        sub_6ch = raw_6ch[::step][:target_seq_len]
        if len(sub_6ch) < target_seq_len:
            pad = np.zeros((target_seq_len - len(sub_6ch), 6))
            sub_6ch = np.vstack([sub_6ch, pad])
            
        X_list.append(sub_6ch)
        y_list.append(w["state_id"])
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def train_lstm_network(
    train_windows: List[Dict],
    val_windows: List[Dict],
    output_dir: str = "models/lstm",
    epochs: int = 15,
    batch_size: int = 64
) -> Tuple[ActivityLSTM, Dict, Dict]:
    """
    Trains the ActivityLSTM network with early stopping and validation tracking.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Training LSTM Deep Sequence Network on Raw Sensor Streams ---")
    
    # 1. Prepare raw sequences
    X_train_raw, y_train = prepare_raw_sensor_windows(train_windows, target_seq_len=50)
    X_val_raw, y_val = prepare_raw_sensor_windows(val_windows, target_seq_len=50)
    
    # 2. Compute channel-wise mean and std exclusively on training set
    ch_mean = np.mean(X_train_raw, axis=(0, 1))
    ch_std = np.std(X_train_raw, axis=(0, 1)) + 1e-6
    
    X_train_norm = (X_train_raw - ch_mean) / ch_std
    X_val_norm = (X_val_raw - ch_mean) / ch_std
    
    # Save channel normalization
    norm_dict = {"mean": ch_mean.tolist(), "std": ch_std.tolist()}
    with open(os.path.join(output_dir, "channel_norm.json"), "w") as f:
        json.dump(norm_dict, f, indent=2)
        
    # Class weights for CrossEntropyLoss to address class imbalance
    class_counts = np.bincount(y_train, minlength=5)
    total_samples = len(y_train)
    weights = total_samples / (5.0 * np.maximum(class_counts, 1.0))
    class_weights_t = torch.tensor(weights, dtype=torch.float32)
    
    train_dataset = SensorSequenceDataset(X_train_norm, y_train)
    val_dataset = SensorSequenceDataset(X_val_norm, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cpu")
    model = ActivityLSTM(input_dim=6, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    best_val_f1 = -1.0
    best_weights = None
    patience = 4
    no_improve_epochs = 0
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            
        train_loss /= len(train_dataset)
        
        # Validation Evaluation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_y.numpy())
                
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        
        # Fall F1 on validation set
        bin_true = (val_true == 3).astype(int)
        bin_pred = (val_preds == 3).astype(int)
        tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
        fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))
        fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))
        val_fall_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Fall F1: {val_fall_f1*100:.2f}%")
        
        if val_fall_f1 > best_val_f1:
            best_val_f1 = val_fall_f1
            best_weights = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  Early stopping triggered at epoch {epoch}")
                break
                
    total_train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Save model weights
    model_path = os.path.join(output_dir, "lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    
    config = {
        "model_name": "Long Short-Term Memory (LSTM)",
        "type": "Comparative Deep Sequence Network",
        "input_dim": 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
        "best_val_fall_f1": float(best_val_f1),
        "train_time_seconds": float(total_train_time)
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    return model, config, norm_dict
