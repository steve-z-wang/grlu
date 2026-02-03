# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRLU (Global Reward Local Updates) is a research project investigating biologically plausible alternatives to backpropagation. It uses evolution-strategy-style perturbations with a global reward signal to train neural networks on MNIST.

**Core algorithm:** Forward pass with noise → compute reward (negative cross-entropy) → antithetic sampling (+noise vs -noise) → local weight update: `ΔW = noise × reward_diff × input`

## Commands

```bash
# Setup
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py

# Train GRLU
python src/train.py --config experiments/exp-1/config.json

# Train backprop baseline
python src/train_backprop.py --config experiments/exp-1/config_backprop.json
```

## Architecture

### Source Files
- `src/model.py` - Pure functional JAX model: `prob_top_k()`, `hard_top_k()`, `forward()`, `forward_eval()`
- `src/train.py` - GRLU training with `grlu_step()` (JIT-compiled), cosine-scheduled LR and noise
- `src/train_backprop.py` - Standard backpropagation baseline for comparison
- `src/download_data.py` / `src/prepare_data.py` - MNIST data pipeline

### Key Concepts
- **Antithetic sampling:** Compare +noise and -noise outcomes to reduce variance
- **Probabilistic top-k WTA:** Gumbel noise for stochastic k-sparse selection during training
- **Cosine annealing:** Both learning rate and noise scale follow cosine schedules

### Config Format
```json
{
  "model": { "layer_sizes": [784, 256, 10] },
  "training": {
    "epochs": 10, "batch_size": 1, "n_perturbations": 128,
    "noise_max": 0.1, "noise_min": 0.0,
    "lr_max": 0.1, "lr_min": 0.0,
    "k": 0.5, "seed": 42
  }
}
```
- `k`: Top-k fraction for sparse activations (null = use ReLU instead)
- `n_perturbations`: Number of noise samples per update (1 = most biologically plausible)

### Data
- Raw MNIST: `data/raw/`
- Processed (normalized): `data/processed/` - X_train.npy, y_train.npy, X_test.npy, y_test.npy

## Research Context

GRLU works well for shallow networks (1-2 hidden layers) but struggles with deep networks due to credit assignment. This aligns with Complementary Learning Systems theory: biology uses slow learning (evolution/ES) for feature extractors and fast learning (Hebbian) for association on pre-trained latent spaces.
