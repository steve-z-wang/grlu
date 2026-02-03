"""
GRLU Model - Pure functional implementation.
"""

import jax
import jax.numpy as jnp


def prob_top_k(Z, k, key):
    """Probabilistic top-k selection using Gumbel noise.

    Higher Z = higher probability of being selected.
    """
    batch, dim = Z.shape
    k_count = jnp.maximum(1, jnp.floor(k * dim).astype(jnp.int32))

    # Add Gumbel noise for stochastic selection
    gumbel_noise = jax.random.gumbel(key, Z.shape)
    perturbed = Z + gumbel_noise

    sorted_desc = jnp.sort(perturbed, axis=1)[:, ::-1]
    k_idx = jnp.maximum(0, k_count - 1)
    idx = jnp.broadcast_to(k_idx, (batch,))[:, None]
    threshold = jnp.take_along_axis(sorted_desc, idx, axis=1)

    mask = (perturbed >= threshold).astype(Z.dtype)
    return mask


def hard_top_k(Z, k):
    """Deterministic top-k selection for evaluation."""
    batch, dim = Z.shape
    k_count = jnp.maximum(1, jnp.floor(k * dim).astype(jnp.int32))

    sorted_desc = jnp.sort(Z, axis=1)[:, ::-1]
    k_idx = jnp.maximum(0, k_count - 1)
    idx = jnp.broadcast_to(k_idx, (batch,))[:, None]
    threshold = jnp.take_along_axis(sorted_desc, idx, axis=1)

    mask = (Z >= threshold).astype(Z.dtype)
    return mask


def init_params(layer_sizes, seed=42):
    """Initialize network parameters.

    Args:
        layer_sizes: List of layer dimensions, e.g. [784, 256, 10]
        seed: Random seed

    Returns:
        params: List of (W, b) tuples
    """
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


def forward(params, X, noises=None, k=None, key=None):
    """Forward pass through network.

    Args:
        params: List of (W, b) tuples
        X: Input (batch, in_dim)
        noises: Optional list of noise tensors per layer
        k: Optional top-k fraction (0-1). If None, uses ReLU.
        key: JAX random key (required if k is provided)

    Returns:
        logits: Output (batch, out_dim)
        activations: List of (input, output) per layer for weight updates
        key: Updated random key (or None)
    """
    activations = []
    for i, (W, b) in enumerate(params):
        # Layer norm
        X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        act_input = X_norm

        # Linear
        Z = X_norm @ W.T + b

        # Activation for hidden layers
        if i < len(params) - 1:
            if k is not None:
                # Probabilistic top-k WTA
                key, subkey = jax.random.split(key)
                mask = prob_top_k(Z, k, subkey)
                out = Z * mask
            else:
                # ReLU
                out = jnp.maximum(0, Z)
            # Add noise after activation
            if noises is not None:
                out = out + noises[i]
            X = out
        else:
            # Output layer - no activation
            if noises is not None:
                Z = Z + noises[i]
            X = Z

        activations.append((act_input, X if i < len(params) - 1 else None))

    return X, activations, key


def forward_eval(params, X, k=None):
    """Forward pass for evaluation (deterministic, no noise)."""
    for i, (W, b) in enumerate(params):
        X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        Z = X_norm @ W.T + b
        if i < len(params) - 1:
            if k is not None:
                # Hard top-k for evaluation
                mask = hard_top_k(Z, k)
                X = Z * mask
            else:
                # ReLU
                X = jnp.maximum(0, Z)
        else:
            X = Z
    return X


def compute_accuracy(params, X, y, k=None):
    """Compute classification accuracy."""
    logits = forward_eval(params, X, k)
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def compute_sparsity(params, X, k=None):
    """Compute sparsity (fraction of inactive neurons) for hidden layers."""
    sparsities = []
    for i, (W, b) in enumerate(params[:-1]):
        X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        Z = X_norm @ W.T + b
        if k is not None:
            mask = hard_top_k(Z, k)
            out = Z * mask
        else:
            out = jnp.maximum(0, Z)
        sparsity = float((out == 0).mean())
        sparsities.append(sparsity)
        X = out
    return sparsities
