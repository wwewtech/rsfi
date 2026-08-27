<div align="center">
  
# RSFI - Riemannian System Fidelity Index

**A Geometric Analysis of One-Class Embedding Guardrails for LLM Jailbreak Detection: Safe-Aware Discriminant Correction and Pooled Within-Class Whitening**

<a href="README.md">⬅️ Back to Language Selection</a>

</div>

---

## On the Name

"Riemannian" is the project's historical name. The full Riemannian machinery (Log/Exp maps) is implemented in `src/rsfi/geometry.py` and covered by unit tests, but the benchmarked pipeline uses Euclidean spherical geometry (L2 + Ledoit-Wolf ZCA + SVD/discriminant directions). The separation between library and experiment is deliberate.

---

## 📌 Abstract

The integration of Large Language Models (LLMs) into critical domains (finance, law, healthcare) is hindered by fundamental vulnerabilities. Direct attacks — *Jailbreaking* and *Prompt Injection* — require filtering at the API gateway level.

Current state-of-the-art methods either require full access to the model's weights (**White-Box** approaches) or introduce critical latency bottlenecks. Classifiers like *Meta Llama Guard* add 100–200 ms of latency per request. Using fast heuristic algorithms based on cosine similarity leads to the issue of **spatial anisotropy** — vectors cluster into a "semantic cone", causing the system to block perfectly safe requests (high False Positive Rate).

**RSFI (Riemannian System Fidelity Index)** introduces a fundamentally different approach. It is a dynamic control algorithm operating at the API gateway level (Black-Box) that detects contextual drift on the fly. The method is based on projecting embeddings into the geometry of a unit hypersphere $\mathbb{S}^{d-1}$, applying Zero Component Analysis (ZCA) Whitening, and performing orthogonal projection in the tangent space $T_S \mathbb{S}^{d-1}$.

---

## 🔥 Key Highlights

* ⚡ **Fast Validation:** Full pipeline cost (embedding + filtering) is ~10-15 ms on CPU.
* 🛡️ **Few-Shot & Black-Box API Protection:** Does not require access to LLM weights. Eliminates the need to fine-tune neural networks (requires 50-200 labeled examples for calibration).
* 📐 **Elimination of the "Semantic Cone":** Utilizing spherical ZCA whitening mathematically realigns correlated vectors, addressing spatial anisotropy.
* 🎯 **Few-Shot Context Detection:** Uses small reference sets (50-200 examples) to detect semantic drift without large-scale training.
* 🪐 **Fast First-Line Defense:** Designed as a fast filtering layer (~0.02ms) to reduce load on slower, more accurate LLM-based judges.

**Important**: RSFI is a **fast geometric filter**, not a replacement for comprehensive guardrails. It trades accuracy for speed,
achieving 0.75-0.85 ROC-AUC vs 0.90-0.95 for fine-tuned transformers. Best used as first-line defense in multi-layer systems.

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

| Parameter / Method | Fine-tuned classifiers (DeBERTa / Llama Guard) | **Discriminant filter $B1$/$B1b$ (this repo)** |
| :--- | :---: | :---: |
| **Model Access** | Black-Box | **Black-Box (embedding API)** |
| **Scoring Latency** | single-digit to tens of ms | **~0.003 ms (single dot product, measured in E7)** |
| **Calibration Memory** | full classifier weights | **1 direction vector + whitening matrix** |
| **ROC-AUC (Wild, 5 seeds)** | **0.841 ± 0.004** (deberta-v2) / **0.724 ± 0.009** (toxic-bert) | **0.867 ± 0.006 ($B1$), LogReg ceiling 0.877** (`data/results/E2d_safe_aware_multidataset.csv`, `E9_external_baselines.csv`) |

**Note on honesty**: earlier claims of AUC = 1.0000 were based on test-set leakage and have been retracted. All current numbers are reproducible from the CSVs in `data/results/`; full methodology is in `docs/RESEARCH_REPORT.md`, and the VAK-level audit summary is in `docs/VAK_VERDICT.md`.

---

## 📂 Repository Architecture

The codebase follows modern Python package standards:

```text
📦 rsfi
 ┣ 📂 src/rsfi          # Core library package
 ┃ ┣ 📜 geometry.py     # RiemannianSphere: geodesic distance, Log/Exp maps on S^(d-1)
 ┃ ┣ 📜 whitening.py    # SphericalWhitening: ZCA whitening with L2 re-projection
 ┃ ┣ 📜 filter.py       # RSFIFilter (1D) & MultiDimensionalRSFIFilter (k-D)
 ┃ ┣ 📜 engine.py       # ProductionSFIEngine & SFIBenchmarkRunner (sentence-transformers)
 ┃ ┗ 📂 datasets        # WildChatBenchmarkRunner (HuggingFace dataset integration)
 ┣ 📂 tests             # Pytest unit & stress test suite
 ┃ ┣ 📜 test_geometry.py    # Riemannian axiom verification
 ┃ ┣ 📜 test_whitening.py   # ZCA fit/transform invariants
 ┃ ┣ 📜 test_filter.py      # Pythagorean decomposition & QR orthonormality
 ┃ ┗ 📜 test_math_advanced.py  # Boundary stress, singularities, rank-deficiency, float precision
 ┣ 📂 experiments        # Runnable experiment scripts (E2-E10)
 ┃ ┗ 📂 archive          # Legacy demos and early evaluation scripts
 ┣ 📂 data               # Experimental outputs
 ┃ ┣ 📂 results          # Benchmark outputs (*.csv)
 ┃ ┣ 📂 reports          # Generated reports
 ┃ ┗ 📂 figures          # ROC curves and latency plots (*.png)
 ┗ 📂 docs               # Documentation
   ┣ 📜 RESEARCH_REPORT.md  # Honest evaluation results (E1-E10)
   ┣ 📜 math.md             # Full mathematical derivations
   ┣ 📜 PROJECT_EVOLUTION.md # Project history and methodology corrections
   ┗ 📂 archive              # Historical drafts and audit logs
```

---

## 🚀 Quick Start & Reproducibility

### Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/wwewtech/rsfi.git
cd rsfi
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run only the advanced mathematical stress suite
pytest tests/test_math_advanced.py -v
```

### Running Experiments

```bash
# Homogeneous dataset evaluation (E2)
python experiments/E2_homogeneous_datasets.py

# Strict FPR operating point analysis (E3)
python experiments/E3_operating_point.py

# External baselines comparison (E7)
python experiments/E7_external_baselines.py

# Sigma_T vs Sigma_W whitening and kNN baselines (E8, Tables 4-5 of the report)
python experiments/E8_sigma_w_whitening.py
```


---

<div align="center">
  <i>This project is distributed under the open <a href="LICENSE">MIT License</a>.</i>
</div>



