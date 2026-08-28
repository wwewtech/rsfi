<div align="center">
  
# RSFI - Riemannian System Fidelity Index

**A Geometric Analysis of One-Class Embedding Guardrails for LLM Jailbreak Detection: Safe-Aware Discriminant Correction and Pooled Within-Class Whitening**

<a href="README.md">⬅️ Back to Language Selection</a> &nbsp; | &nbsp; <a href="README_RU.md">🇷🇺 Русская Версия</a>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-185%20Passed-brightgreen.svg)]()
[![Latency Sub-Millisecond](https://img.shields.io/badge/Latency-~0.003%20ms%20(GPU)-orange.svg)]()
[![VAK Readiness](https://img.shields.io/badge/VAK%20Readiness-K2%20Verified-blueviolet.svg)]()

</div>

---

## 📖 On the Name and Scope

"Riemannian System Fidelity Index" is the project's historical name. The full Riemannian machinery (Logarithmic and Exponential maps on $\mathbb{S}^{d-1}$) is implemented in the library (`src/rsfi/geometry.py`, class `RiemannianSphere`) and covered by unit tests (`tests/test_geometry.py`).

However, the **benchmarked experimental pipeline** (all results in [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)) uses Euclidean spherical geometry: $L_2$-normalization, analytical Ledoit-Wolf ZCA whitening, and SVD/discriminant directions. The separation between the universal mathematical library and the empirical benchmark pipeline is strict and deliberate.

---

## 📌 Abstract

The deployment of Large Language Models (LLMs) in critical enterprise systems (fintech, healthcare, legal tech) is vulnerable to adversarial prompt attacks (*Jailbreak*, *Prompt Injection*) and semantic drift.

Current defense methods either require internal model activation access (**White-Box** approaches) or introduce critical latency bottlenecks (**100–300 ms** per request for LLM judges like Meta Llama Guard). Fast heuristics based on cosine similarity suffer from **spatial anisotropy** (the "semantic cone" problem), leading to unacceptably high False Positive Rates (FPR).

**RSFI** investigates the representation geometry of safety and introduces a family of ultra-fast Black-Box filters for L1 API gateways:
1. **Safe-Aware Discriminant Correction ($B1$):** Subtracting the safe class centroid ($w = \mu_{mal} - \mu_{safe}$) eliminates the blindness of one-class filters to benign contexts, improving ROC-AUC on the heterogeneous WildChat dataset by **+7.7…+10.5 percentage points** ($p < 0.0001$, DeLong test).
2. **Within-Class Whitening ($\Sigma_W$):** We formally prove the covariance decomposition $\Sigma_T = \Sigma_W + \Sigma_B$. Whitening with the within-class covariance $\Sigma_W^{-1/2}$ eliminates local anisotropy without compressing the between-class margin, recovering **43.9%** of the accuracy loss caused by total whitening on average.
3. **Sub-Millisecond Scoring:** Inference requires only a single dot product ($O(d)$, **~0.003 ms** on GPU), outperforming fine-tuned transformer baselines (*ProtectAI DeBERTa-v3*, *unitary/toxic-bert*) across all three benchmark datasets.

---

## 🔥 Key Highlights

* ⚡ **Sub-Millisecond Latency:** Vector scoring takes **~0.003 ms** (GPU) / **~0.02 ms** (CPU). The complete end-to-end pipeline with local ONNX embedding operates within 5–10 ms.
* 🛡️ **Few-Shot & Black-Box:** Requires no access to LLM weights or internal activations. Calibrated with just 50–200 labeled examples without fine-tuning neural networks.
* 📐 **Mathematical Rigor:** Formal proof of covariance decomposition $\Sigma_T = \Sigma_W + \Sigma_B$; analytical Ledoit-Wolf shrinkage avoids singularity when $N_{ref} < d$.
* 🏆 **Outperforms Heavy Classifiers:** On real-world WildChat dialogues, method $B1$ achieves **ROC-AUC = 0.867–0.875**, outperforming *ProtectAI DeBERTa-v3* (0.841) and *toxic-bert* (0.724).
* 🪐 **Defense-in-Depth:** Serves as an ideal **L1 Fast-path** pre-filter before heavy L2/L3 guardrails, intercepting 17–33% of attacks at strict $\text{FPR} \le 1\%$.

---

## 📐 Mathematical Foundation

<details>
<summary><b>Click to expand the mathematical model</b></summary>
<br>

### 1. Safe-Aware Discriminant Direction (Method $B1$)
For calibration sets of malicious $X_{mal}$ and benign $X_{safe}$ prompts, empirical centroids $\mu_{mal}$ and $\mu_{safe}$ are computed. The difference vector defines the separating direction:
$$ w = \mu_{mal} - \mu_{safe}, \quad \hat{w} = \frac{w}{\|w\|_2} $$
For a new query $x$ with normalized embedding $\hat{x} = E(x)/\|E(x)\|_2$, the risk score is evaluated in $O(d)$:
$$ s_{B1}(x) = \langle \hat{x}, \hat{w} \rangle $$

### 2. Covariance Decomposition & Within-Class Whitening ($\Sigma_W$)
The total covariance matrix $\Sigma_T$ of the pooled calibration data decomposes into within-class $\Sigma_W$ and between-class $\Sigma_B$:
$$ \Sigma_T = \Sigma_W + \Sigma_B = \frac{1}{2}(\Sigma_{mal} + \Sigma_{safe}) + \frac{1}{4}(\mu_{mal} - \mu_{safe})(\mu_{mal} - \mu_{safe})^T $$
Total whitening $\Sigma_T^{-1/2}$ compresses the leading eigenvalue of $\Sigma_B$, suppressing discriminability on heterogeneous data. Within-class whitening $\mathbf{W} = \Sigma_W^{-1/2}$ (with Ledoit-Wolf regularization) eliminates local anisotropy while preserving the separation margin:
$$ \hat{x}_w = \frac{\Sigma_W^{-1/2}(\hat{x} - \mu)}{\|\Sigma_W^{-1/2}(\hat{x} - \mu)\|_2}, \quad s_{B1w}(x) = \langle \hat{x}_w, \hat{w}_w \rangle $$

### 3. Riemannian Geometry on $\mathbb{S}^{d-1}$
Normalized vectors lie on the smooth compact Riemannian manifold $\mathbb{S}^{d-1}$. The Riemannian logarithmic map $\text{Log}_S(y)$ maps vector $y$ isometrically into the flat tangent space $T_S \mathbb{S}^{d-1}$ at the system prompt anchor $S$:
$$ \mathbf{v} = \text{Log}_S(\mathbf{y}) = \frac{\theta}{\sin \theta} \bigl(\mathbf{y} - S \cos \theta\bigr), \quad \theta = \arccos(\langle S, \mathbf{y} \rangle) $$

</details>

---

## 📊 State-of-the-Art Comparison (ROC-AUC across 3 Datasets, 5 Seeds, mean ± std)

All values are strictly verified by CSV logs in `data/results/` and covered by 185 automated tests:

| Method / Model | Type / Memory | Scoring Latency | Wild (Heterogeneous) | ToxicChat (Homogeneous) | XSTest (Homonymy) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ProtectAI DeBERTa-v3** | Fine-tuned Transformer | 8.80 ms (GPU) | 0.8405 ± 0.0036 | 0.5698 ± 0.0100 | 0.4022 ± 0.0123 |
| **unitary/toxic-bert** | Fine-tuned Transformer | 4.20 ms (GPU) | 0.7236 ± 0.0087 | 0.7934 ± 0.0126 | 0.6416 ± 0.0178 |
| **A1 Naive Cosine (raw)** | 1 vector, 1-class | **~0.003 ms** | 0.7846 ± 0.0039 | 0.9158 ± 0.0044 | 0.7618 ± 0.0211 |
| **A2 RSFI-SVD ($k=20$, raw)**| $k$ vectors, 1-class | 0.005 ms | 0.7875 ± 0.0080 | 0.9382 ± 0.0056 | 0.8463 ± 0.0253 |
| **B1 Safe-Aware Disc (raw)** | 1 vector, 2-class | **~0.003 ms** | **0.8668 ± 0.0060** | **0.9509 ± 0.0048** | 0.7851 ± 0.0203 |
| **B1b Disc ($\Sigma_T$-whitened)** | 1 vector + ZCA | **~0.003 ms** | 0.8297 ± 0.0090 | **0.9617 ± 0.0020** | **0.8970 ± 0.0132** |
| **B1w Disc ($\Sigma_W$-whitened)** | 1 vector + ZCA | **~0.003 ms** | **0.8475 ± 0.0071** | **0.9680 ± 0.0016** | **0.8999 ± 0.0128** |
| **Qwen3-8B (4096d) B1 / B1w** | 1 vector (4096d) | **~0.004 ms** | **0.8752 ± 0.0055** | **0.9632 ± 0.0028** | **0.8272 ± 0.0229** |
| **C1 LogReg (Supervised ceiling)**| 1 vector + bias | ~0.003 ms | 0.8766 ± 0.0041 | 0.9702 ± 0.0037 | 0.8542 ± 0.0152 |

> **Scientific Honesty Note**: Earlier claims of perfect accuracy (ROC-AUC = 1.0000) were caused by test-set leakage into ZCA calibration and have been retracted. All current numbers are reproducible from CSV logs in `data/results/`. Full methodology, VAK K2 readiness verdict, and audit history are in [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md).

---

## 📂 Repository Architecture

```text
📦 rsfi
 ┣ 📂 src/rsfi          # Core library package
 ┃ ┣ 📜 geometry.py     # RiemannianSphere: geodesic distance, Log/Exp maps on S^(d-1)
 ┃ ┣ 📜 whitening.py    # SphericalWhitening: ZCA whitening with Ledoit-Wolf shrinkage
 ┃ ┣ 📜 filter.py       # RSFIFilter (1D) & MultiDimensionalRSFIFilter (k-D)
 ┃ ┣ 📜 engine.py       # ProductionSFIEngine & SFIBenchmarkRunner
 ┃ ┗ 📂 datasets        # Dataset loaders and stream processing modules
 ┣ 📂 tests             # Test suite (200 passed)
 ┃ ┣ 📜 test_report_consistency.py # 200 automated consistency tests for Tables 1–10
 ┃ ┣ 📜 test_geometry.py    # Riemannian axiom verification
 ┃ ┣ 📜 test_whitening.py   # ZCA fit/transform invariants
 ┃ ┣ 📜 test_filter.py      # Pythagorean decomposition & QR orthonormality
 ┃ ┗ 📜 test_math_advanced.py  # Boundary stress, singularities, rank-deficiency
 ┣ 📂 experiments        # Executable experiment scripts (E1–E10, E6b, E6c, E8, E9, E9b)
 ┃ ┣ 📜 E2d_safe_aware_multidataset.py  # Multi-dataset Safe-Aware battery
 ┃ ┣ 📜 E2e_qwen_extension.py           # Qwen3-8B (4096d) multi-dataset evaluation
 ┃ ┣ 📜 E8_sigma_w_whitening.py         # Within-class Sigma_W whitening
 ┃ ┣ 📜 E9_external_baselines.py        # Comparison with DeBERTa-v3 and Toxic-BERT
 ┃ ┣ 📜 E6b_obfuscation_boundary.py     # Obfuscation degradation map (1620 rows)
 ┃ ┣ 📜 E6c_defense_aware_adaptive_attack.py # Defense-aware adaptive adversary (E6c)
 ┃ ┗ 📜 E9b_external_obfuscation.py     # External classifier obfuscation (180 rows)
 ┣ 📂 data               # Data and experimental outputs
 ┃ ┣ 📂 results          # Benchmark output CSVs with per-seed metrics
 ┃ ┣ 📂 reports          # Generated reports
 ┃ ┗ 📂 figures          # ROC curves and latency plots (*.png)
 ┗ 📂 docs               # Documentation
   ┣ 📜 RESEARCH_REPORT.md  # Comprehensive research report (Tables 1–9), VAK K2 verdict, audit chronicle
   ┣ 📜 math.md             # Theoretical derivations and semantic geometry
   ┣ 📜 ARCHITECTURE.md     # System architecture (Rust API Gateway), Three-Clocks, EU AI Act
   ┣ 📜 source.md           # 55 verified academic and industry sources
   ┗ 📂 audit_history       # Historical audit logs and protocols
```

---

## 🚀 Quick Start & Reproducibility

### Requirements
- Python 3.10+
- PyTorch, sentence-transformers, scikit-learn, scipy

### Installation

```bash
git clone https://github.com/wwewtech/rsfi.git
cd rsfi
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
# Run full suite of 185 tests (including consistency validation for all report tables)
pytest tests/ -v

# Run only advanced mathematical stress tests
pytest tests/test_math_advanced.py -v
```

### Running Key Experiments

```bash
# 1. Multi-model Safe-Aware battery (Tables 1–2 of the report)
python experiments/E2d_safe_aware_multidataset.py

# 2. Within-class Sigma_W whitening and k-NN (Tables 4–5 of the report)
python experiments/E8_sigma_w_whitening.py

# 3. Comparison with DeBERTa-v3 and Toxic-BERT (Table 7 of the report)
python experiments/E9_external_baselines.py

# 4. Obfuscation boundary evaluation (Tables 8–9 of the report)
python experiments/E6b_obfuscation_boundary.py
python experiments/E9b_external_obfuscation.py
```

---

<div align="center">
  <i>This project is distributed under the open <a href="LICENSE">MIT License</a>.</i>
</div>
