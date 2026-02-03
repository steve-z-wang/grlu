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

---

## Run 5: ReLU (128 perturbations)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 1
- n_perturbations: 128
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: None (ReLU)
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc | Sparsity |
|-------|-----------|----------|----------|
| 1     | 0.9170    | 0.9189   | 48%      |
| 2     | 0.9180    | 0.9268   | 51%      |
| 3     | 0.9360    | 0.9317   | 51%      |
| 4     | 0.9440    | 0.9390   | 51%      |
| 5     | 0.9460    | 0.9418   | 51%      |
| 6     | 0.9470    | 0.9428   | 51%      |
| 7     | 0.9470    | 0.9438   | 51%      |
| 8     | 0.9480    | 0.9441   | 51%      |
| 9     | 0.9470    | 0.9445   | 51%      |
| 10    | 0.9470    | 0.9445   | 51%      |

**Final:** Test accuracy 94.45%

---

## Run 6: Probabilistic Top-K WTA (128 perturbations, k=0.5)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 1
- n_perturbations: 128
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.5
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc | Sparsity |
|-------|-----------|----------|----------|
| 1     | 0.9180    | 0.9193   | 50%      |
| 2     | 0.9370    | 0.9320   | 50%      |
| 3     | 0.9370    | 0.9342   | 50%      |
| 4     | 0.9410    | 0.9340   | 50%      |
| 5     | 0.9360    | 0.9376   | 50%      |
| 6     | 0.9400    | 0.9409   | 50%      |
| 7     | 0.9410    | 0.9421   | 50%      |
| 8     | 0.9410    | 0.9412   | 50%      |
| 9     | 0.9410    | 0.9413   | 50%      |
| 10    | 0.9410    | 0.9413   | 50%      |

**Final:** Test accuracy 94.13%

---

## Run 7: Probabilistic Top-K WTA (128 perturbations, k=0.4)

**Settings:**
- layer_sizes: [784, 256, 10]
- epochs: 10
- batch_size: 1
- n_perturbations: 128
- noise_max: 0.1
- noise_min: 0.0
- lr_max: 0.1
- lr_min: 0.0
- k: 0.4
- seed: 42

**Results:**
| Epoch | Train Acc | Test Acc | Sparsity |
|-------|-----------|----------|----------|
| 1     | 0.9140    | 0.9189   | 60%      |
| 2     | 0.9370    | 0.9329   | 60%      |
| 3     | 0.9400    | 0.9327   | 60%      |
| 4     | 0.9400    | 0.9330   | 60%      |
| 5     | 0.9400    | 0.9361   | 60%      |
| 6     | 0.9410    | 0.9375   | 60%      |
| 7     | 0.9440    | 0.9401   | 60%      |
| 8     | 0.9450    | 0.9402   | 60%      |
| 9     | 0.9450    | 0.9403   | 60%      |
| 10    | 0.9450    | 0.9403   | 60%      |

**Final:** Test accuracy 94.03%
