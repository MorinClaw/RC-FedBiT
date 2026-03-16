# RC-FedBiT: Rate-Controlled Federated Bit-width Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **RC-FedBiT** is a communication-efficient federated learning framework that adapts gradient bit-width to wireless channel conditions via rank-1 compression and channel-aware rate selection.

## Overview

Federated learning over wireless networks suffers from severe communication bottlenecks. RC-FedBiT addresses this by combining:

1. **Rank-1 Gradient Compression** — Each client compresses weight updates into a binary matrix **B** plus two low-precision vectors (h1, h2), dramatically reducing communication cost.
2. **Channel-Adaptive Rate Selection (CA-RS)** — Based on per-client Rayleigh-faded SNR, the server dynamically selects payload precision (FP32 > INT8 > binary-only) to maximize accuracy under varying channel quality.
3. **Noise-Impaired Aggregation & Channel Variance Adaptation (NIA-CVA)** — Aggregation accounts for channel noise and gradient variance misalignment across heterogeneous clients.
4. **Adaptive Frequency Calibration (AFC)** — Post-aggregation calibration corrects spectral drift introduced by compression and quantization.

### Architecture

```
Client i
+-- Local training (SGD, local_epochs=5)
+-- Compute delta_W = W_local - W_global
+-- Rank-1 compress: delta_W ~= B x (h1 x h2)
+-- Channel-adaptive payload selection (FP32/INT8/binary)
        |  Rayleigh channel (SNR ~ Exponential)
        v
Server
+-- Weighted aggregation of decompressed gradients
+-- NIA-CVA alignment
+-- AFC calibration -> global model update
```

## Results

### Main Comparison (CIFAR-10, ViT-Tiny, 100 clients, 10% participation, 100 rounds)

| Method | Final Acc (%) | Comm Cost |
|--------|:------------:|:---------:|
| SignSGD | 23.26 | Low |
| PowerSGD | 24.10 | Medium |
| **RC-FedBiT** | **26.29** | **Low** |
| FedAvg | 26.54 | High (FP32) |
| QSGD | 28.06 | Medium |

RC-FedBiT achieves accuracy competitive with FedAvg while using significantly lower communication bandwidth, outperforming all low-communication baselines.

### Ablation Study (CIFAR-10, 100 rounds)

| Variant | Components | Final Acc (%) |
|---------|-----------|:------------:|
| A2: CA-RS only | Rank-1 + Channel-Adaptive Rate Selection | 27.13 |
| A3: NIA-CVA | Rank-1 + CA-RS + NIA-CVA | 27.13 |
| A4: Full RC-FedBiT | All components (+ AFC) | 27.01 |

> Note: Channel robustness (SNR sweep), Non-IID, CIFAR-100, and scalability results are in progress and will be added upon completion.

## Requirements

```bash
pip install torch torchvision timm numpy
```

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1.

## Quick Start

### 1. Run Main Comparison

```bash
python experiments/run_main.py --method rc_fedbit --rounds 100
python experiments/run_main.py --method fedavg --rounds 100
python experiments/run_main.py --method qsgd --rounds 100
python experiments/run_main.py --method signsgd --rounds 100
python experiments/run_main.py --method powersgd --rounds 100
```

### 2. Ablation Study

```bash
# A2: Channel-aware rate scaling only
python experiments/run_ablation.py --variant A2 --rounds 100
# A3: NIA-CVA without AFC
python experiments/run_ablation.py --variant A3 --rounds 100
# A4: Full model
python experiments/run_ablation.py --variant A4 --rounds 100
```

### 3. Channel Robustness

```bash
python experiments/run_channel.py --snr 0 --method all --rounds 100
python experiments/run_channel.py --snr 7 --method all --rounds 100
python experiments/run_channel.py --snr 15 --method all --rounds 100
```

### 4. Non-IID Heterogeneity

```bash
python experiments/run_noniid.py --alpha 0.1 --rounds 100
python experiments/run_noniid.py --alpha 0.5 --rounds 100
```

## Project Structure

```
RC-FedBiT/
+-- src/
|   +-- compression/
|   |   +-- rank1_compress.py      # Rank-1 gradient compressor (GPU SVD + binary quant)
|   +-- channel/
|   |   +-- channel_adaptive.py    # Channel-adaptive selector + Rayleigh simulator
|   +-- federated/
|   |   +-- client.py              # FedBiT client (local training + compression)
|   |   +-- server.py              # FedBiT server (aggregation + AFC calibration)
|   +-- baselines/
|   |   +-- fedavg.py
|   |   +-- signsgd.py
|   |   +-- qsgd.py
|   |   +-- powersgd.py
|   +-- data/
|       +-- partition.py           # Dirichlet / IID data partitioning
+-- experiments/
|   +-- run_main.py                # Main comparison experiments
|   +-- run_ablation.py            # Ablation variants (A0-A4)
|   +-- run_channel.py             # Channel SNR robustness
|   +-- run_noniid.py              # Non-IID heterogeneity
|   +-- run_multi.py               # Multi-dataset / scalability
+-- results/                       # JSON result files (gitignored)
+-- README.md
+-- LICENSE
```

## Method Details

### Rank-1 Gradient Compression

For each layer weight delta W in R^{m x n}:

```
delta_W ~= B * (h1 outer h2)
```

where B = sign(delta_W) is a 1-bit matrix, and h1 in R^m, h2 in R^n are computed via GPU power iteration (3 iterations).
Compression ratio: `32mn / (mn + 32(m+n))`.

### Channel-Adaptive Rate Selection

Given client SNR gamma and cosine-annealed thresholds gamma_high(t), gamma_low(t):

| Channel Quality | Payload | Overhead |
|----------------|---------|----------|
| gamma > gamma_high | B + h1/h2 in FP32 | 100% |
| gamma_low < gamma <= gamma_high | B + h1/h2 in INT8 | 85% |
| gamma <= gamma_low | B only | 50% |

## Citation

```bibtex
@article{wang2025rcfedbit,
  title={RC-FedBiT: Rate-Controlled Federated Learning with Bit-width Adaptation},
  author={Wang, Maolin and others},
  year={2025},
  note={Preprint}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
