"""
Backprop Training on MNIST (for comparison with GRLU)
"""

import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_data():
    """Load processed MNIST data."""
    X_train = jnp.array(np.load(os.path.join(DATA_DIR, "X_train.npy")))
    y_train = jnp.array(np.load(os.path.join(DATA_DIR, "y_train.npy")))
    X_test = jnp.array(np.load(os.path.join(DATA_DIR, "X_test.npy")))
    y_test = jnp.array(np.load(os.path.join(DATA_DIR, "y_test.npy")))
    return X_train, y_train, X_test, y_test


def load_config(config_path):
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def init_params(layer_sizes, seed=42):
    """Initialize parameters."""
    key = jax.random.PRNGKey(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        scale = jnp.sqrt(2.0 / in_dim)
        W = jax.random.normal(subkey, (out_dim, in_dim)) * scale
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params


def forward(params, X):
    """Forward pass."""
    for i, (W, b) in enumerate(params):
        # Layer norm
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        Z = X @ W.T + b
        if i < len(params) - 1:
            X = jnp.maximum(0, Z)  # ReLU
        else:
            X = Z
    return X


def cross_entropy_loss(params, X, y):
    """Cross-entropy loss."""
    logits = forward(params, X)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = jnp.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    log_probs = jnp.log(probs[jnp.arange(logits.shape[0]), y] + 1e-10)
    return -log_probs.mean()


def compute_accuracy(params, X, y):
    """Compute accuracy."""
    logits = forward(params, X)
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def compute_sparsity(params, X):
    """Compute sparsity for hidden layers."""
    sparsities = []
    for i, (W, b) in enumerate(params[:-1]):
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        Z = X @ W.T + b
        sparsity = float((Z < 0).mean())
        sparsities.append(sparsity)
        X = jnp.maximum(0, Z)
    return sparsities


# JIT compiled gradient step
@jax.jit
def backprop_step(params, X, y, lr):
    """Single backprop training step."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, X, y)

    new_params = []
    for (W, b), (dW, db) in zip(params, grads):
        new_W = W - lr * dW
        new_b = b - lr * db
        new_params.append((new_W, new_b))

    return new_params, loss


def train(config, output_dir=None):
    """Train MLP with backprop on MNIST."""
    # Extract config
    layer_sizes = config['model']['layer_sizes']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr_max = config['training']['lr_max']
    lr_min = config['training']['lr_min']
    seed = config['training']['seed']

    key = jax.random.PRNGKey(seed)

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize model
    params = init_params(layer_sizes, seed=seed)
    print(f"Model: {' -> '.join(map(str, layer_sizes))}")

    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Cosine schedule
        cosine = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        lr = lr_min + (lr_max - lr_min) * cosine

        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        total_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1:3d}", leave=False)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            params, loss = backprop_step(params, X_batch, y_batch, jnp.array(lr))
            total_loss += float(loss)
            pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

        # Evaluate
        train_acc = compute_accuracy(params, X_train[:1000], y_train[:1000])
        test_acc = compute_accuracy(params, X_test, y_test)
        sparsities = compute_sparsity(params, X_test[:1000])
        sparsity_str = " | ".join([f"L{i+1}: {s:.0%}" for i, s in enumerate(sparsities)])
        print(f"Epoch {epoch + 1:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | {sparsity_str}")

    # Save model
    if output_dir:
        model_path = os.path.join(output_dir, 'model_backprop.npz')
    else:
        model_path = 'model_backprop.npz'

    weights = {f'W{i}': np.array(W) for i, (W, b) in enumerate(params)}
    weights.update({f'b{i}': np.array(b) for i, (W, b) in enumerate(params)})
    np.savez(model_path, **weights)
    print(f"Model saved to {model_path}")

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with Backprop on MNIST')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for model')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, output_dir=args.output)
