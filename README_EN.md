<div align="center">
  
# RSFI - Riemannian System Fidelity Index

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

| Parameter / Method | Representation Eng. | Meta Llama Guard 4 | SAFENUDGE / ZEDD | **RSFI (Our Method)** |
| :--- | :---: | :---: | :---: | :---: |
| **Model Access** | White-Box (Required) | Black-Box | Black-Box | **Black-Box (API)** |
| **Latency Overhead** | ~0 ms | > 100–200 ms | < 15 ms | **~0.021 ms (< 10 ms)** |
| **Fine-Tuning Required** | No | Yes | No | **Few-Shot** |
| **Anisotropy Mitigation** | N/A | N/A | Partial | **Full ZCA Whitening** |
| **Separability (ROC-AUC)**| 0.92 | 0.96 | 0.89 | **0.856** (4096d) |

**Note on Performance**: Previous claims of 1.0000 AUC were based on methodologically flawed experiments with test set leakage. 
Honest evaluation (experiments E1-E10) shows RSFI achieves **0.75-0.85 ROC-AUC** on real-world data, competitive with 
fast geometric methods but below fine-tuned transformers (~0.90-0.95). See `docs/RESEARCH_REPORT.md` for details.

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
```


---

<div align="center">
  <i>This project is distributed under the open <a href="LICENSE">MIT License</a>.</i>
</div>



