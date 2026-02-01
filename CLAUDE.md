# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository investigating Global Reward Local Updates (GRLU) - an alternative to backpropagation that uses evolution-strategy-style perturbations with a global reward signal for neural network training.

## Commands

Run MNIST training:
```bash
python train_mnist.py
```

Dependencies: JAX, scikit-learn (for data loading), numpy

## Architecture

Three-file structure:
- `layer.py` - `Layer` class with GRLU update mechanism
- `normalization.py` - Layer normalization function
- `train_mnist.py` - `MLP` class, training loop, and MNIST data loading

### GRLU Algorithm

1. Forward pass with original weights → get reward
2. Forward pass with noise-perturbed weights → get perturbed reward
3. Compute `reward_diff = perturbed - original`
4. Update: `delta = correlation * noise * reward_diff` (correlation = outer product of layer outputs and inputs)
5. Homeostatic scaling: normalize each neuron's weights to std=1

Key design choices:
- Reward = negative cross-entropy loss
- Layer normalization applied before each layer (biological gain control analog)
- Multiple perturbations per batch for stability (default: 10)
- Activity-based correlation allows small weights to grow
