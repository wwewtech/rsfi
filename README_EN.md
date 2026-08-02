<div align="center">
  
# 🛡️ RSFI: Riemannian System Fidelity Index 🌐

**A Method for Dynamic Control of Semantic Drift and Sycophancy in Large Language Models Based on Non-Euclidean Geometry**

<a href="README.md">⬅️ Back to Language Selection</a>

</div>

---

## 📌 Abstract

The integration of Large Language Models (LLMs) into critical domains (finance, law, healthcare) is hindered by fundamental vulnerabilities. Neural networks are prone to **sycophancy** — easily aligning with malicious or erroneous user context. Defending them against direct attacks like *Jailbreaking* and *Prompt Injection* often requires complex workarounds.

Current state-of-the-art methods either require full access to the model's weights (**White-Box** approaches) or introduce critical latency bottlenecks. Classifiers like *Meta Llama Guard* add 100–200 ms of latency per request. Using fast heuristic algorithms based on cosine similarity leads to the issue of **spatial anisotropy** — vectors cluster into a "semantic cone", causing the system to block perfectly safe requests (high False Positive Rate).

**RSFI (Riemannian System Fidelity Index)** introduces a fundamentally different approach. It is a dynamic control algorithm operating at the API gateway level (Black-Box) that detects contextual drift on the fly. The method is based on projecting embeddings into the geometry of a unit hypersphere $\mathbb{S}^{d-1}$, applying Zero Component Analysis (ZCA) Whitening, and performing orthogonal projection in the tangent space $T_S \mathbb{S}^{d-1}$.

---

## 🔥 Key Highlights

* ⚡ **Ultra-Fast Validation (< 10 ms):** Calculating the index for a single vector takes only **21 microseconds**, allowing up to 47,000 checks per second. Inference speed remains entirely unaffected.
* 🛡️ **Zero-Shot & Black-Box API Protection:** Does not require access to LLM weights. Eliminates the need to fine-tune third-party discriminator classifiers.
* 📐 **Elimination of the "Semantic Cone":** Utilizing spherical ZCA whitening mathematically realigns correlated vectors, fully resolving spatial anisotropy.
* 🎯 **Precision Context Decoupling:** Employs Gram-Schmidt orthogonalization to surgically separate the prompt's useful semantic payload from potential threats.
* 🪐 **Zero-Day Attack Detection:** Projecting vectors into a $k$-dimensional orthonormal subspace enables the blocking of completely novel, unseen attack patterns.

---

## 📐 Mathematical Foundation

<details>
<summary><b>Click to expand the RSFI mathematical model</b></summary>
<br>

**1. Spherical Whitening (ZCA Whitening)**
We transform the embedding space to eliminate anisotropy while preserving the original vector orientation:
$$ \mathbf{y} = \mathbf{W}_{zca} (\mathbf{E} - \boldsymbol{\mu}), \quad \mathbf{W}_{zca} = \mathbf{U} \boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T $$

**2. Mapping to Tangent Space $T_S \mathbb{S}^{d-1}$**
We apply the Riemannian logarithmic map $\text{Log}_S(\mathbf{y})$ at the system anchor point $S$ (safety instructions):
$$ \mathbf{v} = \text{Log}_S(\mathbf{y}) = \frac{\theta}{\sin \theta} \bigl(\mathbf{y} - S \cos \theta\bigr), \quad \text{where } \theta = \arccos(\langle S, \mathbf{y} \rangle) $$

**3. Constructing the Orthogonal Threat Basis**
Using the Gram-Schmidt process, we isolate the threat projection orthogonal to the system rule:
$$ \mathbf{e}_{thr} = \frac{\mathbf{v}_{thr} - \langle \mathbf{v}_{thr}, \mathbf{e}_{sys} \rangle \mathbf{e}_{sys}}{\|\mathbf{v}_{thr} - \langle \mathbf{v}_{thr}, \mathbf{e}_{sys} \rangle \mathbf{e}_{sys}\|} $$

**4. RSFI Objective Function**
We calculate the final system fidelity index:
$$ \text{RSFI}(r) = \pi_{sys}(r) - \lambda \cdot \pi_{thr}(r) $$
*If $\text{RSFI}(r)$ drops below the defined threshold $\tau$, the API gateway preemptively terminates generation.*

</details>

---

## 📊 State-of-the-Art Comparison

| Parameter / Method | Representation Eng. | Meta Llama Guard 4 | SAFENUDGE / ZEDD | **RSFI (Our Method)** |
| :--- | :---: | :---: | :---: | :---: |
| **Model Access** | White-Box (Required) | Black-Box | Black-Box | **Black-Box (API)** |
| **Latency Overhead** | ~0 ms | > 100–200 ms | < 15 ms | **~0.021 ms (< 10 ms)** |
| **Fine-Tuning Required** | No | Yes | No | **Zero-Shot** |
| **Anisotropy Mitigation** | N/A | N/A | Partial | **Full ZCA Whitening** |
| **Separability (ROC-AUC)**| 0.92 | 0.96 | 0.89 | **1.0000** |

---

## 📂 Repository Architecture

The entire codebase is structured according to modern Python package standards and Open Science principles:

```text
📦 rsfi
 ┣ 📂 data           # Datasets and experimental data
 ┃ ┣ 📂 results      # Benchmark outputs (*.csv)
 ┃ ┗ 📂 telemetry    # Telemetry reports and logs (*.json)
 ┣ 📂 docs           # Academic documentation and paper drafts
 ┃ ┣ 📂 analogues    # Detailed analysis and comparison of existing methods
 ┃ ┣ 📜 math.md      # Riemannian geometry derivations and RSFI formulas
 ┃ ┗ 📜 ...          # Russian VAK drafts, architecture logs, Open Science designs
 ┣ 📂 figures        # Graphical artifacts, ROC curves, and latency plots (*.png)
 ┗ 📂 src            # Source code and algorithms
   ┣ 📂 analysis     # Evaluation scripts and LLM Judge integrations
   ┣ 📂 benchmarks   # Execution scripts for JailbreakBench and Wild-10k
   ┣ 📂 tests        # Unit tests for hypothesis validation and ZCA verification
   ┗ 📂 utils        # Auxiliary report generators
```

---

## 🚀 Quick Start & Reproducibility

### Requirements
- Python 3.10+
- `numpy`, `scipy`, `scikit-learn`, `matplotlib`

### Installation

```bash
git clone https://github.com/wwewtech/rsfi.git
cd rsfi
pip install numpy scipy scikit-learn matplotlib
```

### Running Core Benchmarks

Executable tests are structured within the unified `src` package:

1. **Comprehensive Math Validation (10 Stages) & Latency Test**:
   ```bash
   python src/tests/test2_aai.py
   ```
2. **Semantic Drift Simulation in Multi-Turn Dialogues**:
   ```bash
   python src/tests/test3_advanced.py
   ```
3. **Zero-Day Attack Blocking via $k$-Dimensional Subspace**:
   ```bash
   python src/tests/test4_subspace.py
   ```

---

## 🎓 Academic Citation

This research strictly adheres to the standards of the Higher Attestation Commission (VAK RF) and the JailbreakBench validation framework (NeurIPS 2024).

If you use the **RSFI** algorithm or materials from this repository in your academic publications, please cite our work:

```bibtex
@article{osanov2026rsfi,
  title={Метод динамического контроля семантического дрейфа и сикофантии языковых моделей на основе неевклидовой геометрии и Риманова индекса системной лояльности (RSFI)},
  author={Osanov, P. V.},
  journal={Bulletin of PSUTI / VAK RF},
  year={2026}
}
```

---

<div align="center">
  <i>This project is distributed under the open <a href="LICENSE">MIT License</a>.</i>
</div>
