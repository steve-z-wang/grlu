"""
GRLU Layer implementation with JAX
"""

import jax
import jax.numpy as jnp

from normalization import layer_norm


class Layer:
    """
    GRLU Layer with local updates and global reward.
    """

    def __init__(self, in_dim: int, out_dim: int, key: jax.random.PRNGKey):
        """Initialize layer with Xavier initialization."""
        scale = jnp.sqrt(2.0 / in_dim)
        self.W = jax.random.normal(key, (out_dim, in_dim)) * scale

        # Cached values from forward pass (private)
        self._last_input = None
        self._last_output = None

    def forward(self, X: jnp.ndarray, activation: str = "relu", noise: jnp.ndarray = None) -> jnp.ndarray:
        """
        Forward pass through layer.

        Args:
            X: Input (batch, in_dim)
            activation: "relu" or "none"
            noise: Optional noise to add to weights (out_dim, in_dim)

        Returns:
            output: (batch, out_dim)
        """
        # Layer normalization (like biological gain control)
        X = layer_norm(X)

        # Cache input (after normalization)
        self._last_input = X

        # Use perturbed weights if noise provided
        W = self.W + noise if noise is not None else self.W

        # Linear transformation
        Z = X @ W.T  # (batch, out_dim)

        # Activation
        if activation == "relu":
            output = jnp.maximum(0, Z)
        else:
            output = Z

        # Cache output
        self._last_output = output

        return output

    def _compute_delta(self, noise: jnp.ndarray, reward_diff: float) -> jnp.ndarray:
        """
        Compute weight update delta.

        delta = correlation * noise * reward_diff

        Where correlation is per-sample (output * input), then averaged.
        Activity-based (not weight-based), so small weights can grow.

        Args:
            noise: Random perturbation (out_dim, in_dim)
            reward_diff: Reward difference (scalar)

        Returns:
            delta: Weight update (out_dim, in_dim)
        """
        # Per-sample correlation, then average
        # For each sample: outer(output, input), then mean over batch
        # einsum: 'bo,bi->oi' means: batch(b), out(o), in(i)
        batch_size = self._last_input.shape[0]
        correlation = jnp.einsum('bo,bi->oi', self._last_output, self._last_input) / batch_size

        # Delta = correlation * noise * reward_diff
        delta = correlation * noise * reward_diff

        return delta

    def _normalize_weights(self):
        """
        Normalize weights per neuron (homeostatic scaling).
        Each neuron's weights normalized to std=1.
        """
        std = self.W.std(axis=1, keepdims=True)
        self.W = self.W / (std + 1e-8)

    def update(self, noise: jnp.ndarray, reward_diff: float, lr: float):
        """
        Apply GRLU update to layer.

        Args:
            noise: Random perturbation (out_dim, in_dim)
            reward_diff: Reward difference (scalar)
            lr: Learning rate
        """
        # Compute delta
        delta = self._compute_delta(noise, reward_diff)

        # Update weights
        self.W = self.W + lr * delta

        # Homeostatic scaling
        self._normalize_weights()
