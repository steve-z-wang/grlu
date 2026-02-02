"""
GRLU Layer implementation with JAX
"""

import jax
import jax.numpy as jnp

from normalization import layer_norm


class Layer:
    """GRLU Layer with local Hebbian updates and global reward signal."""

    def __init__(self, in_dim: int, out_dim: int, key: jax.random.PRNGKey):
        """Initialize layer with Xavier initialization."""
        scale = jnp.sqrt(2.0 / in_dim)
        self.W = jax.random.normal(key, (out_dim, in_dim)) * scale
        self.out_dim = out_dim
        self.in_dim = in_dim

        # Cached values for Hebbian learning
        self._last_input = None
        self._last_output = None
        self._last_sparsity = 0.0

    def forward(self, X: jnp.ndarray, activation: str = "relu", noise: jnp.ndarray = None) -> jnp.ndarray:
        """
        Forward pass through layer.

        Args:
            X: Input (batch, in_dim)
            activation: "relu" or "none"
            noise: Optional noise to add to weights for ES perturbation

        Returns:
            output: (batch, out_dim)
        """
        # Layer normalization (biological gain control)
        X = layer_norm(X)

        self._last_input = X

        # Apply weight perturbation if provided
        W = self.W + noise if noise is not None else self.W

        # Linear transformation
        Z = X @ W.T

        # Activation
        if activation == "relu":
            output = jnp.maximum(0, Z)
        else:
            output = Z

        self._last_output = output
        return output

    def compute_sparsity(self):
        """Compute sparsity stats (call only when needed)."""
        if self._last_output is not None:
            self._last_sparsity = float((self._last_output == 0).mean())
        return self._last_sparsity

    def _normalize_weights(self):
        """Normalize weights per neuron (homeostatic scaling)."""
        std = self.W.std(axis=1, keepdims=True)
        self.W = self.W / (std + 1e-8)

    def update(self, noise: jnp.ndarray, reward_diff: float, lr: float):
        """
        Apply GRLU update: ES gradient * Hebbian correlation.

        Args:
            noise: Random perturbation used in ES
            reward_diff: Reward difference from perturbation
            lr: Learning rate
        """
        # Hebbian correlation: outer(output, input)
        batch_size = self._last_input.shape[0]
        correlation = jnp.einsum('bo,bi->oi', self._last_output, self._last_input) / batch_size

        # GRLU update: ES gradient gated by Hebbian correlation
        delta = noise * reward_diff * correlation

        # Update weights
        self.W = self.W + lr * delta

        # Homeostatic scaling (prevents weight explosion)
        self._normalize_weights()
