"""
GRLU Training on MNIST

Global Reward, Local Updates - a biologically plausible learning algorithm
that combines Evolution Strategy gradient estimation with Hebbian learning.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from layer import Layer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def load_data():
    """Load processed MNIST data."""
    X_train = jnp.array(np.load(os.path.join(PROCESSED_DIR, "X_train.npy")))
    y_train = jnp.array(np.load(os.path.join(PROCESSED_DIR, "y_train.npy")))
    X_test = jnp.array(np.load(os.path.join(PROCESSED_DIR, "X_test.npy")))
    y_test = jnp.array(np.load(os.path.join(PROCESSED_DIR, "y_test.npy")))
    return X_train, y_train, X_test, y_test


class MLP:
    """Simple MLP with GRLU layers."""

    def __init__(self, layer_sizes: list, key: jax.random.PRNGKey):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], subkey)
            self.layers.append(layer)

    def forward(self, X: jnp.ndarray, noises: list = None) -> jnp.ndarray:
        """Forward pass through all layers."""
        for i, layer in enumerate(self.layers):
            activation = "relu" if i < len(self.layers) - 1 else "none"
            noise = noises[i] if noises is not None else None
            X = layer.forward(X, activation=activation, noise=noise)
        return X


def softmax(x):
    """Stable softmax."""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, y):
    """Cross-entropy loss."""
    probs = softmax(logits)
    batch_size = logits.shape[0]
    log_probs = jnp.log(probs[jnp.arange(batch_size), y] + 1e-10)
    return -log_probs.mean()


def compute_reward(model, X, y, noises=None):
    """Compute reward (negative loss)."""
    logits = model.forward(X, noises=noises)
    loss = cross_entropy_loss(logits, y)
    return -loss


def compute_accuracy(model, X, y):
    """Compute accuracy."""
    logits = model.forward(X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def grlu_step(model, X, y, key, noise_scale=0.1, lr=0.1, n_perturbations=10):
    """
    GRLU training step:
    1. Compute baseline reward
    2. Try n perturbations, accumulate ES gradients
    3. Apply update with Hebbian gating
    """
    reward_orig = compute_reward(model, X, y)

    # Accumulate gradients over perturbations
    accumulated_deltas = [jnp.zeros_like(layer.W) for layer in model.layers]

    for _ in range(n_perturbations):
        # Generate noise for each layer
        noises = []
        for layer in model.layers:
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, layer.W.shape) * noise_scale
            noises.append(noise)

        # Compute perturbed reward
        reward_pert = compute_reward(model, X, y, noises=noises)
        reward_diff = float(reward_pert - reward_orig)

        # Accumulate ES gradient
        for i, noise in enumerate(noises):
            accumulated_deltas[i] = accumulated_deltas[i] + noise * reward_diff

    # Forward pass to cache activations for Hebbian term
    _ = model.forward(X)

    # Apply averaged update with Hebbian gating
    for i, layer in enumerate(model.layers):
        batch_size = layer._last_input.shape[0]
        correlation = jnp.einsum('bo,bi->oi', layer._last_output, layer._last_input) / batch_size

        # Average ES gradient * Hebbian gate
        delta = (accumulated_deltas[i] / n_perturbations) * correlation

        layer.W = layer.W + lr * delta
        layer._normalize_weights()

    return float(reward_orig), key


def train(
    epochs: int = 100,
    batch_size: int = 128,
    noise_scale: float = 0.1,
    lr_max: float = 0.1,
    lr_min: float = 0.01,
    n_perturbations: int = 10,
    seed: int = 42
):
    """Train MLP with GRLU on MNIST."""
    key = jax.random.PRNGKey(seed)

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize model
    key, subkey = jax.random.split(key)
    model = MLP([784, 256, 10], subkey)
    print("Model: 784 -> 256 -> 10")

    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Cosine learning rate schedule
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs))

        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        total_reward = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1:3d}", leave=False)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            reward, key = grlu_step(
                model, X_batch, y_batch, key,
                noise_scale=noise_scale, lr=lr, n_perturbations=n_perturbations
            )
            total_reward += reward

            pbar.set_postfix(reward=f"{total_reward / (batch_idx + 1):.4f}")

        # Evaluate
        train_acc = float(compute_accuracy(model, X_train[:1000], y_train[:1000]))
        test_acc = float(compute_accuracy(model, X_test, y_test))

        # Get sparsity for hidden layer
        sparsity = model.layers[0].compute_sparsity()
        print(f"Epoch {epoch + 1:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Sparsity: {sparsity:.0%}")

    return model


if __name__ == "__main__":
    model = train(
        epochs=100,
        batch_size=128,
        noise_scale=0.1,
        lr_max=0.1,
        lr_min=0.01,
        n_perturbations=10,
        seed=42
    )
