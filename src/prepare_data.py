"""
Prepare MNIST data: normalize and split into train/test
"""

import os
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def prepare_mnist():
    """Load raw data, normalize, split, and save to data/processed/"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw
    print("Loading raw data...")
    X = np.load(os.path.join(RAW_DIR, "X.npy"))
    y = np.load(os.path.join(RAW_DIR, "y.npy"))

    # Normalize: scale to [0,1], then standardize with MNIST stats
    print("Normalizing...")
    X = (X / 255.0 - 0.1307) / 0.3081

    # Train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Save processed data
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"Saved to {PROCESSED_DIR}/")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  y_train.npy: {y_train.shape}")
    print(f"  X_test.npy: {X_test.shape}")
    print(f"  y_test.npy: {y_test.shape}")


if __name__ == "__main__":
    prepare_mnist()
