"""
Normalization functions for GRLU
"""

import jax.numpy as jnp


def layer_norm(X: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Layer normalization (per-sample, across features).

    Like biological gain control - normalizes input range.

    Args:
        X: Input (batch, features)
        eps: Small constant for numerical stability

    Returns:
        Normalized X with mean=0, std=1 per sample
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + eps)
