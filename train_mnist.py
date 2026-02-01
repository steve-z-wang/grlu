"""
GRLU Training on MNIST
"""

import jax
import jax.numpy as jnp
from layer import Layer


# ============================================================
# Data Loading
# ============================================================

def load_mnist():
    """Load MNIST dataset."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    except:
        from tensorflow.keras.datasets import mnist as mnist_keras
        (X_train, y_train), (X_test, y_test) = mnist_keras.load_data()
        X = jnp.vstack([X_train.reshape(-1, 784), X_test.reshape(-1, 784)]).astype('float32')
        y = jnp.hstack([y_train, y_test]).astype('int')

    # Normalize: mean=0, std=1
    X = (X / 255.0 - 0.1307) / 0.3081

    # Convert to jax arrays
    X = jnp.array(X)
    y = jnp.array(y)

    # Train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, y_train, X_test, y_test


# ============================================================
# Model
# ============================================================

class MLP:
    """Simple MLP with GRLU layers."""

    def __init__(self, layer_sizes: list, key: jax.random.PRNGKey):
        """
        Args:
            layer_sizes: e.g. [784, 256, 10]
            key: JAX random key
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], subkey)
            self.layers.append(layer)

    def forward(self, X: jnp.ndarray, noises: list = None) -> jnp.ndarray:
        """
        Forward pass through all layers.

        Args:
            X: Input (batch, in_dim)
            noises: Optional list of noise for each layer
        """
        for i, layer in enumerate(self.layers):
            activation = "relu" if i < len(self.layers) - 1 else "none"
            noise = noises[i] if noises is not None else None
            X = layer.forward(X, activation=activation, noise=noise)
        return X

    def update(self, noises: list, reward_diff: float, lr: float):
        """Update all layers with GRLU."""
        for layer, noise in zip(self.layers, noises):
            layer.update(noise, reward_diff, lr)


# ============================================================
# Loss and Reward
# ============================================================

def _softmax(x):
    """Stable softmax."""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def _cross_entropy_loss(logits, y):
    """Cross-entropy loss."""
    probs = _softmax(logits)
    batch_size = logits.shape[0]
    log_probs = jnp.log(probs[jnp.arange(batch_size), y] + 1e-10)
    return -log_probs.mean()


def compute_reward(model, X, y, noises=None):
    """Compute reward (negative loss)."""
    logits = model.forward(X, noises=noises)
    loss = _cross_entropy_loss(logits, y)
    return -loss  # Higher is better


def compute_accuracy(model, X, y):
    """Compute accuracy."""
    logits = model.forward(X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


# ============================================================
# GRLU Training Step
# ============================================================

def grlu_step(model, X, y, key, noise_scale=0.01, lr=0.01):
    """
    One GRLU update step:
    1. Forward with original weights, get reward
    2. Forward with noisy weights, get reward
    3. Compute reward diff
    4. Update based on noise * reward_diff
    """
    # Original reward (forward pass caches activations)
    reward_orig = compute_reward(model, X, y)

    # Generate noise for each layer
    noises = []
    for layer in model.layers:
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, layer.W.shape) * noise_scale
        noises.append(noise)

    # Perturbed reward (no side effects on weights)
    reward_pert = compute_reward(model, X, y, noises=noises)

    # Reward difference
    reward_diff = reward_pert - reward_orig

    # Forward pass again without noise to cache correct activations for update
    _ = model.forward(X)

    # Apply GRLU update
    model.update(noises, reward_diff, lr)

    return reward_orig, reward_diff, key


# ============================================================
# Training Loop
# ============================================================

def train(
    epochs: int = 50,
    batch_size: int = 128,
    noise_scale: float = 0.01,
    lr: float = 0.01,
    num_perturbations: int = 10,
    seed: int = 42
):
    """Train MLP with GRLU on MNIST."""

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Load data
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize model
    key, subkey = jax.random.split(key)
    model = MLP([784, 256, 10], subkey)
    print("Model: 784 -> 256 -> 10")

    # Training
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        total_reward = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Multiple perturbations for stability
            for _ in range(num_perturbations):
                reward, reward_diff, key = grlu_step(
                    model, X_batch, y_batch, key,
                    noise_scale=noise_scale, lr=lr
                )
                total_reward += float(reward)

        # Evaluate
        train_acc = float(compute_accuracy(model, X_train[:1000], y_train[:1000]))
        test_acc = float(compute_accuracy(model, X_test, y_test))
        avg_reward = total_reward / (n_batches * num_perturbations)

        print(f"Epoch {epoch + 1:3d} | Reward: {avg_reward:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model = train(
        epochs=50,
        batch_size=128,
        noise_scale=0.01,
        lr=0.01,
        num_perturbations=10,
        seed=42
    )
