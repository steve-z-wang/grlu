# GRLU Design

## Update Mechanism

### ES-Style Perturbation with Conservation

**Step 1: Forward pass with current weights**
```
reward_original = compute_reward(W, input)
```

**Step 2: Generate noise (mutation)**
```
noise = random_normal(shape of W)
```

**Step 3: Forward pass with perturbed weights**
```
reward_mutated = compute_reward(W + noise, input)
```

**Step 4: Compute raw update**
```
reward_diff = reward_mutated - reward_original
delta_W = noise * reward_diff
```

**Step 5: Conservation (per-neuron normalization)**
```
# Subtract mean per neuron's inputs (each row)
delta_W = delta_W - delta_W.mean(axis=1, keepdims=True)
```

**Step 6: Apply update**
```
W += learning_rate * delta_W
```

## Why This Works

1. **Contrastive comparison** - Compare original vs mutated to get positive/negative signal
2. **Noise as exploration** - Like DNA mutation, try variations
3. **Credit assignment is local** - Each weight uses its own noise × global reward_diff
4. **Conservation prevents runaway** - Per-neuron mean subtraction ensures zero-sum updates
5. **GPU-friendly** - All matrix operations, fully parallelizable

## MNIST Implementation

### Architecture
- Input: 784 (28×28 flattened, normalized to mean=0, std=1)
- Hidden: 256 (ReLU activation)
- Output: 10 (softmax)

### Input Normalization
```
X = (X - 0.1307) / 0.3081  # MNIST standard values
```

### Global Reward
- Negative cross-entropy loss (higher = better)
