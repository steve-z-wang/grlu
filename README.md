# Global Reward Local Updates (GRLU)

Research repository investigating GRLU - a biologically plausible alternative to backpropagation using evolution-strategy-style perturbations with a global reward signal.

## Algorithm

GRLU uses noise perturbations and reward correlation to update weights locally:

1. Forward pass with noise added to neuron outputs
2. Compute reward (negative cross-entropy loss)
3. Use antithetic sampling: compare +noise vs -noise outcomes
4. Update weights: `delta_W = noise * reward_diff * input`

Key features:
- **No backpropagation** - gradients don't flow backwards
- **Local learning rule** - each layer updates based on local information + global reward
- **Biologically plausible** - similar to how dopamine modulates synaptic plasticity

## Results

On MNIST with architecture 784 → 256 → 10 (10 epochs, batch size 1):

| Perturbations | Test Accuracy | Notes |
|---------------|---------------|-------|
| 1 | ~90% | More biologically plausible (single trial) |
| 16 | 92.83% | Multiple samples reduce variance |
| 128 | 94.45% | High sample count approaches gradient quality |
| 128 | ~98% | 2 hidden layers with skip connections |

The single-perturbation result is more interesting from a biological perspective - the brain doesn't average over 128 parallel perturbations. Higher perturbation counts essentially approximate the gradient, losing biological plausibility.

See `experiments/exp-1/configs.md` for detailed results.

## Key Findings

### What works:
- **Shallow networks** (1-2 hidden layers)
- **Small batch sizes** (batch=1 best for online learning)
- **Multiple perturbations** (16 helps stabilize learning)
- **Probabilistic top-k WTA** for sparse activations

### What doesn't work:
- **Deep networks** - credit assignment problem; reward signal gets diluted
- **Extreme sparsity** (k < 0.05) - not enough active neurons for information flow
- **Shared weights / recurrence** - conflicting gradients from different time steps

## Limitations & Insights

GRLU struggles with deep networks because it lacks a mechanism to assign credit across many layers. This led to an important insight:

**The brain uses complementary learning systems:**
- **Slow learning** (neocortex): Evolved over millions of years, creates feature extractors
- **Fast learning** (hippocampus): Hebbian-style, operates on pre-learned latent space

A single algorithm cannot do both. Hebbian/GRLU is suited for fast learning on top of a pre-trained feature space, not for learning features from scratch.

**Key insight:** Biology uses neither backprop for either system:

| System | Biological Algorithm | Timescale |
|--------|---------------------|-----------|
| Feature extraction (vision) | **Evolution / ES** | Millions of years |
| Fast memory (hippocampus) | **Hebbian** | Milliseconds |

Evolution is essentially ES - random mutations (noise) + selection pressure (reward). The visual cortex was optimized this way over evolutionary time, then Hebbian learning handles fast online adaptation. A fully bio-plausible system could use ES for feature learning + Hebbian for fast association - no backprop anywhere.

## Future Direction

A hybrid approach:
```
[Autoencoder backbone] → 128-dim latent → [Hebbian/GRLU] → output
   (trained offline)        (frozen)       (fast online learning)
```

This mirrors biological learning:
- Autoencoder = visual cortex (slow, offline optimization)
- Hebbian layer = hippocampus (fast, online association learning)

**See [slow-fast-learn](https://github.com/steve-z-wang/slow-fast-learn)** for an implementation of this hybrid approach.

## Related Work

Our findings align with **Complementary Learning Systems (CLS)** theory from cognitive neuroscience:

| System | Brain Region | Learning | Representations |
|--------|-------------|----------|-----------------|
| Slow | Neocortex | Gradual, statistical | Overlapping, distributed |
| Fast | Hippocampus | Rapid, episodic | Sparse, separated |

**Key papers:**
- [Why There Are Complementary Learning Systems (McClelland et al., 1995)](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) - Foundational CLS theory
- [Complementary Learning Systems (2011)](https://pubmed.ncbi.nlm.nih.gov/22141588/) - Updated framework
- [Hybrid Neural Networks for Continual Learning (2025)](https://www.nature.com/articles/s41467-025-56405-9) - Recent work combining ANNs with spiking networks inspired by corticohippocampal circuits

The CLS theory suggests that a single learning algorithm cannot efficiently do both feature extraction and fast memory formation - exactly what we observed with GRLU struggling on deep networks but working well as a shallow associative learner.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare data
python src/download_data.py
python src/prepare_data.py

# Train
python src/train.py --config experiments/exp-1/config.json
```

## Files

- `src/model.py` - GRLU model (forward pass, top-k sparsity)
- `src/train.py` - Training loop with antithetic sampling
- `experiments/` - Experiment configs and results
