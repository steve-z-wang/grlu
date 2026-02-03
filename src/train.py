"""
GRLU Training on MNIST

Biologically plausible learning using:
- Output noise (neural firing variability)
- Antithetic sampling (+/- noise)
- Hebbian-like credit assignment: ΔW = (noise × reward_diff) ⊗ input
"""

import os
import json
import argparse
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from model import init_params, forward, compute_accuracy, compute_sparsity

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


@partial(jax.jit, static_argnums=(6,))
def grlu_step(params, X, y, key, noise_scale, lr, n_perturbations, k=None):
    """
    Single GRLU training step with multiple perturbations.

    1. Try n_perturbations different noise samples in parallel
    2. Antithetic sampling: compare +noise vs -noise for each
    3. Average the updates across perturbations
    """
    batch_size = X.shape[0]
    n_layers = len(params)

    # Tile input for parallel perturbations: (batch, dim) -> (batch * n_pert, dim)
    X_tiled = jnp.tile(X, (n_perturbations, 1))
    y_tiled = jnp.tile(y, (n_perturbations,))
    effective_batch = batch_size * n_perturbations

    # Generate output noise for each layer
    noises = []
    for i in range(n_layers):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (effective_batch, params[i][0].shape[0])) * noise_scale
        noises.append(noise)

    # Antithetic sampling: +noise and -noise
    key, subkey = jax.random.split(key)
    logits_pos, activations_pos, _ = forward(params, X_tiled, noises, k, subkey)
    logits_neg, activations_neg, _ = forward(params, X_tiled, [-n for n in noises], k, subkey)

    # Per-sample rewards (negative cross-entropy)
    logits_pos_stable = logits_pos - logits_pos.max(axis=1, keepdims=True)
    exp_pos = jnp.exp(logits_pos_stable)
    probs_pos = exp_pos / exp_pos.sum(axis=1, keepdims=True)
    reward_pos = jnp.log(probs_pos[jnp.arange(effective_batch), y_tiled] + 1e-10)

    logits_neg_stable = logits_neg - logits_neg.max(axis=1, keepdims=True)
    exp_neg = jnp.exp(logits_neg_stable)
    probs_neg = exp_neg / exp_neg.sum(axis=1, keepdims=True)
    reward_neg = jnp.log(probs_neg[jnp.arange(effective_batch), y_tiled] + 1e-10)

    reward_diff = (reward_pos - reward_neg) / 2.0

    # Update each layer: ΔW = (noise × reward_diff) ⊗ input
    new_params = []
    for i in range(n_layers):
        W, b = params[i]
        act_input_pos, _ = activations_pos[i]
        act_input_neg, _ = activations_neg[i]
        act_input = (act_input_pos + act_input_neg) / 2.0

        delta_W = jnp.einsum('bo,bi->oi', noises[i] * reward_diff[:, None], act_input) / n_perturbations
        delta_b = (noises[i] * reward_diff[:, None]).sum(axis=0) / n_perturbations

        new_W = W + lr * delta_W
        new_b = b + lr * delta_b
        new_params.append((new_W, new_b))

    reward_avg = (reward_pos.mean() + reward_neg.mean()) / 2.0
    return new_params, reward_avg, key


def train(config, output_dir=None):
    """Train with GRLU on MNIST."""
    # Extract config
    layer_sizes = config['model']['layer_sizes']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    n_perturbations = config['training'].get('n_perturbations', 1)
    noise_max = config['training']['noise_max']
    noise_min = config['training']['noise_min']
    lr_max = config['training']['lr_max']
    lr_min = config['training']['lr_min']
    k = config['training'].get('k', None)  # None = ReLU, float = top-k
    seed = config['training']['seed']

    key = jax.random.PRNGKey(seed)

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize params
    params = init_params(layer_sizes, seed=seed)
    activation = f"top-k={k}" if k else "ReLU"
    print(f"Model: {' -> '.join(map(str, layer_sizes))} | {activation} | Perturbations: {n_perturbations}")

    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Cosine schedule
        cosine = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        lr = lr_min + (lr_max - lr_min) * cosine
        noise_scale = noise_min + (noise_max - noise_min) * cosine

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

            k_val = jnp.array(k) if k else None
            params, reward, key = grlu_step(
                params, X_batch, y_batch, key,
                jnp.array(noise_scale), jnp.array(lr), n_perturbations, k_val
            )
            total_reward += float(reward)
            pbar.set_postfix(reward=f"{total_reward / (batch_idx + 1):.4f}")

        # Evaluate
        train_acc = compute_accuracy(params, X_train[:1000], y_train[:1000], k)
        test_acc = compute_accuracy(params, X_test, y_test, k)
        sparsities = compute_sparsity(params, X_test[:1000], k)
        sparsity_str = " | ".join([f"L{i+1} sparsity: {s:.0%}" for i, s in enumerate(sparsities)])
        print(f"Epoch {epoch + 1:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | {sparsity_str}")

    # Save model
    if output_dir:
        model_path = os.path.join(output_dir, 'model.npz')
    else:
        model_path = 'model.npz'

    weights = {f'W{i}': np.array(W) for i, (W, b) in enumerate(params)}
    weights.update({f'b{i}': np.array(b) for i, (W, b) in enumerate(params)})
    np.savez(model_path, **weights)
    print(f"Model saved to {model_path}")

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GRLU on MNIST')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for model')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, output_dir=args.output)
