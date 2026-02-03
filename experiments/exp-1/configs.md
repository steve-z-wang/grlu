# Experiment 1 Results

## Run 1: Probabilistic Top-K WTA

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 1
- n_perturbations: 16
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.5
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1     | 0.9060    | 0.9090   |
| 2     | 0.9100    | 0.9178   |
| 3     | 0.9120    | 0.9211   |
| 4     | 0.9220    | 0.9264   |
| 5     | 0.9250    | 0.9239   |
| 6     | 0.9300    | 0.9282   |
| 7     | 0.9290    | 0.9274   |
| 8     | 0.9310    | 0.9281   |
| 9     | 0.9300    | 0.9283   |
| 10    | 0.9300    | 0.9283   |

**Final:** Test accuracy 92.83%

---

## Run 2: Probabilistic Top-K WTA (batch 32, k=0.25)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 32
- n_perturbations: 16
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.25
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1     | 0.8540    | 0.8560   |
| 2     | 0.8730    | 0.8824   |
| 3     | 0.8850    | 0.8909   |
| 4     | 0.8780    | 0.8955   |
| 5     | 0.8850    | 0.8968   |
| 6     | 0.8850    | 0.8971   |
| 7     | 0.8870    | 0.8972   |
| 8     | 0.8870    | 0.8967   |
| 9     | 0.8870    | 0.8969   |
| 10    | 0.8870    | 0.8969   |

**Final:** Test accuracy 89.69%

---

## Run 3: Probabilistic Top-K WTA (batch 16, k=0.25)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 16
- n_perturbations: 16
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.25
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1     | 0.8710    | 0.8793   |
| 2     | 0.8840    | 0.8928   |
| 3     | 0.8970    | 0.9013   |
| 4     | 0.9060    | 0.9054   |
| 5     | 0.9110    | 0.9081   |
| 6     | 0.9040    | 0.9083   |
| 7     | 0.9060    | 0.9083   |
| 8     | 0.9070    | 0.9084   |
| 9     | 0.9070    | 0.9085   |
| 10    | 0.9070    | 0.9085   |

**Final:** Test accuracy 90.85%

---

## Run 4: Probabilistic Top-K WTA (batch 1, k=0.25)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 1
- n_perturbations: 16
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.25
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1     | 0.8950    | 0.9015   |
| 2     | 0.9040    | 0.9084   |
| 3     | 0.9110    | 0.9122   |
| 4     | 0.9160    | 0.9175   |
| 5     | 0.9150    | 0.9225   |
| 6     | 0.9250    | 0.9241   |
| 7     | 0.9280    | 0.9253   |
| 8     | 0.9280    | 0.9254   |
| 9     | 0.9280    | 0.9252   |
| 10    | 0.9280    | 0.9252   |

**Final:** Test accuracy 92.52%
