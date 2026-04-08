# data_loader.py - COMPLETE WORKING VERSION
import wfdb
import numpy as np
import os
import urllib.request

DATA_PATH = "data/mit-bih/"
WINDOW_SIZE = 187

RECORD_NAMES = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

def download_dataset():
    print("📥 Downloading MIT-BIH dataset...")
    os.makedirs(DATA_PATH, exist_ok=True)
    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    for record in RECORD_NAMES:
        print(f"  Downloading {record}...")
        for ext in ['.dat', '.hea', '.atr']:
            url = f"{base_url}{record}{ext}"
            filename = f"{DATA_PATH}{record}{ext}"
            try:
                urllib.request.urlretrieve(url, filename)
            except:
                pass
    print("✅ Download complete!")

def load_local_data():
    print("📂 Loading data...")
    signals = []
    annotations = []
    for record in RECORD_NAMES:
        try:
            if os.path.exists(f"{DATA_PATH}{record}.dat"):
                signal = wfdb.rdrecord(f"{DATA_PATH}{record}")
                ann = wfdb.rdann(f"{DATA_PATH}{record}", 'atr')
                signals.append(signal)
                annotations.append(ann)
                print(f"  ✓ Loaded record {record}")
        except Exception as e:
            print(f"  ✗ Could not load {record}")
    print(f"✅ Loaded {len(signals)} records")
    return signals, annotations

def create_dataset(signals, annotations):
    print("🔄 Extracting heartbeats...")
    all_heartbeats = []
    all_labels = []
    
    total_beats = 0
    half_before = WINDOW_SIZE // 2
    half_after = WINDOW_SIZE - half_before - 1
    
    for i in range(len(signals)):
        signal = signals[i]
        ann = annotations[i]
        
        ecg_signal = signal.p_signal[:, 0]
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
        
        r_peaks = ann.sample
        symbols = ann.symbol
        
        record_beats = 0
        
        for idx, peak in enumerate(r_peaks):
            if peak < half_before or peak > len(ecg_signal) - half_after:
                continue
            
            start = peak - half_before
            end = peak + half_after + 1
            heartbeat = ecg_signal[start:end]
            
            if len(heartbeat) != WINDOW_SIZE:
                continue
            
            symbol = symbols[idx]
            label = 0 if symbol == 'N' else 1
            
            all_heartbeats.append(heartbeat)
            all_labels.append(label)
            record_beats += 1
            total_beats += 1
        
        print(f"  Record {i+1}: {record_beats} beats extracted")
    
    print(f"\n📊 Total beats extracted: {total_beats}")
    
    if len(all_heartbeats) == 0:
        print("❌ No heartbeats extracted!")
        return np.array([]), np.array([])
    
    X = np.array(all_heartbeats)
    y = np.array(all_labels)
    
    print(f"\n✅ Final dataset:")
    print(f"   Total heartbeats: {len(X)}")
    print(f"   Normal beats: {np.sum(y == 0)}")
    print(f"   Abnormal beats: {np.sum(y == 1)}")
    
    X = X.reshape(len(X), WINDOW_SIZE, 1)
    
    return X, y