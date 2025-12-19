import glob
import os
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = "data/raw/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"

# Horizon mapping for FI-2010 labels:
# 0 -> k=10 (Next 10 ticks) | 1 -> k=20 | 2 -> k=30 | 3 -> k=50 | 4 -> k=100
HORIZON_IDX = 0  # We will predict next 10 ticks (k=10)
WINDOW_SIZE = 100  # Sequence length for the Transformer


def compute_normalization(orderbook: np.ndarray, train_frac_for_stats: float = 0.7):
    """Compute mean/std from training portion only"""
    N, F = orderbook.shape
    train_raw_end = int(train_frac_for_stats * N)
    train_slice = orderbook[:train_raw_end]

    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def make_window_labels_balanced(orderbook: np.ndarray, window_size: int, horizon: int, up_percentile: float = 67,
                                down_percentile: float = 33):
    """
    Create BALANCED labels using percentiles instead of fixed thresholds.
    This ensures roughly equal class distribution.

    Args:
        orderbook: Raw LOB data (N, F)
        window_size: Length of input sequence
        horizon: How far ahead to predict
        up_percentile: Percentile for up class (default 67 = top 33%)
        down_percentile: Percentile for down class (default 33 = bottom 33%)

    Returns:
        seq_labels: Array of class labels (0=down, 1=stationary, 2=up)
    """
    N, F = orderbook.shape

    # Compute mid-price
    best_ask = orderbook[:, 0]
    best_bid = orderbook[:, 2]
    mid = 0.5 * (best_ask + best_bid) / 10000.0

    M = N - window_size - horizon + 1
    if M <= 0:
        raise ValueError("window_size + horizon too large")

    t0 = np.arange(window_size - 1, window_size - 1 + M)
    t_future = t0 + horizon
    diff = mid[t_future] - mid[t0]

    # Use percentiles to create balanced classes
    up_threshold = np.percentile(diff, up_percentile)
    down_threshold = np.percentile(diff, down_percentile)

    seq_labels = np.ones(M, dtype=np.int64)  # Default: stationary (1)
    seq_labels[diff > up_threshold] = 2  # Up
    seq_labels[diff < down_threshold] = 0  # Down

    # Print distribution
    unique, counts = np.unique(seq_labels, return_counts=True)
    print(f"\n{'=' * 60}")
    print("BALANCED LABEL DISTRIBUTION")
    print(f"{'=' * 60}")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c:,} ({100 * c / M:.1f}%)")
    print(f"  Up threshold: {up_threshold:.6f}")
    print(f"  Down threshold: {down_threshold:.6f}")
    print(f"{'=' * 60}\n")

    return seq_labels


def build_split_indices(num_windows: int, train_frac: float = 0.7, val_frac: float = 0.15, mode: str = "chronological",
                        seed: int = 42):
    """Split dataset into train/val/test"""
    M = num_windows
    train_end = int(train_frac * M)
    val_end = int((train_frac + val_frac) * M)

    if mode == "chronological":
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(train_end, val_end, dtype=np.int64)
        test_idx = np.arange(val_end, M, dtype=np.int64)
    elif mode == "random":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(M)
        train_idx = perm[:train_end]
        val_idx = perm[train_end:val_end]
        test_idx = perm[val_end:]
    else:
        raise ValueError("mode must be 'chronological' or 'random'")

    return train_idx, val_idx, test_idx


def load_fi2010_partition(root_dir, partition):
    """
    Loads and concatenates all .txt files for a specific partition (Training/Testing).
    Returns transposed features (N, 144) and labels (N,).
    """
    # Construct path: .../NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/
    sub_folder = f"NoAuction_Zscore_{partition}"  # e.g. NoAuction_Zscore_Training
    search_path = os.path.join(root_dir, sub_folder, "*.txt")

    print(f"Searching for files in: {search_path}")

    files = sorted(glob.glob(search_path))
    if not files:
        raise ValueError(f"No .txt files found in {search_path}. Check your path!")

    print(f"Loading {len(files)} files from {sub_folder}...")

    all_features = []
    all_labels = []

    for f_path in files:
        try:
            raw_mat = np.loadtxt(f_path)  # Shape: (149, T)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

        data_T = raw_mat.T
        features = data_T[:, :144]
        labels_all_k = data_T[:, 144:]
        y = labels_all_k[:, HORIZON_IDX] - 1

        all_features.append(features)
        all_labels.append(y)

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    Y = np.concatenate(all_labels, axis=0).astype(np.int64)

    return X, Y


def process_data():
    print("Processing FI-2010 Dataset...")

    X_train_raw, Y_train_raw = load_fi2010_partition(DATA_ROOT, "Training")
    X_test_raw, Y_test_raw = load_fi2010_partition(DATA_ROOT, "Testing")

    split_idx = int(len(X_train_raw) * 0.8)

    X_train = X_train_raw[:split_idx]
    Y_train = Y_train_raw[:split_idx]

    X_val = X_train_raw[split_idx:]
    Y_val = Y_train_raw[split_idx:]

    X_test = X_test_raw
    Y_test = Y_test_raw

    all_lob = np.concatenate([X_train, X_val, X_test], axis=0)
    all_labels = np.concatenate([Y_train, Y_val, Y_test], axis=0)

    offset_train = 0
    offset_val = len(X_train)
    offset_test = len(X_train) + len(X_val)

    def get_valid_indices(start_offset, length, window_size):
        return np.arange(start_offset, start_offset + length - window_size)

    train_indices = get_valid_indices(offset_train, len(X_train), WINDOW_SIZE)
    val_indices = get_valid_indices(offset_val, len(X_val), WINDOW_SIZE)
    test_indices = get_valid_indices(offset_test, len(X_test), WINDOW_SIZE)

    print(f"\nFinal Shapes:")
    print(f"LOB Tensor: {all_lob.shape}")
    print(f"Train Indices: {len(train_indices)}")
    print(f"Val Indices:   {len(val_indices)}")
    print(f"Test Indices:  {len(test_indices)}")

    OUT_PATH = "data/processed/fi2010_processed.pt"
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    obj = {
        "lob": torch.from_numpy(all_lob),
        "seq_labels": torch.from_numpy(all_labels),
        "window_size": WINDOW_SIZE,
        "horizon": 10,
        "train_indices": torch.from_numpy(train_indices),
        "val_indices": torch.from_numpy(val_indices),
        "test_indices": torch.from_numpy(test_indices),
        "mean": torch.zeros(144),
        "std": torch.ones(144),
    }

    torch.save(obj, OUT_PATH)
    print(f"Saved processed dataset to: {OUT_PATH}")


if __name__ == "__main__":
    process_data()
