"""
GRLU Model - Pure functional implementation.
"""

import jax
import jax.numpy as jnp


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


def forward(params, X, noises=None):
    """Forward pass through network with residual connections.

    Args:
        params: List of (W, b) tuples
        X: Input (batch, in_dim)
        noises: Optional list of noise tensors per layer

    Returns:
        logits: Output (batch, out_dim)
        activations: List of (input, output) per layer for weight updates
    """
    activations = []
    for i, (W, b) in enumerate(params):
        # Layer norm
        X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        act_input = X_norm

        # Linear
        Z = X_norm @ W.T + b

        # Activation (ReLU for hidden, none for output)
        if i < len(params) - 1:
            out = jnp.maximum(0, Z)
            # Add noise after activation
            if noises is not None:
                out = out + noises[i]
            X = out
        else:
            # Output layer
            if noises is not None:
                Z = Z + noises[i]
            X = Z

        activations.append((act_input, X if i < len(params) - 1 else None))

    return X, activations


def compute_accuracy(params, X, y):
    """Compute classification accuracy."""
    logits, _ = forward(params, X)
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def compute_sparsity(params, X):
    """Compute sparsity (fraction of inactive neurons) for hidden layers."""
    _, activations = forward(params, X)
    sparsities = []
    for i, (_, output) in enumerate(activations[:-1]):
        if output is not None:
            sparsity = float((output == 0).mean())
            sparsities.append(sparsity)
    return sparsities
