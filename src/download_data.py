"""
Download MNIST dataset to data/raw/
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def download_mnist():
    """Download MNIST and save to data/raw/"""
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        from sklearn.datasets import fetch_openml
        print("Downloading MNIST via sklearn...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    except Exception as e:
        print(f"sklearn failed: {e}")
        print("Trying tensorflow...")
        from tensorflow.keras.datasets import mnist as mnist_keras
        (X_train, y_train), (X_test, y_test) = mnist_keras.load_data()
        X = np.vstack([X_train.reshape(-1, 784), X_test.reshape(-1, 784)]).astype('float32')
        y = np.hstack([y_train, y_test]).astype('int')

    # Save raw data
    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)

    print(f"Saved to {DATA_DIR}/")
    print(f"  X.npy: {X.shape}")
    print(f"  y.npy: {y.shape}")


if __name__ == "__main__":
    download_mnist()
