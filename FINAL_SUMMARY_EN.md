# 📊 Comprehensive Analysis Report: RSFI Project Status

**Analysis Date:** August 5, 2026  
**Analyzer:** Claude Opus 5  
**Scope:** 30 commits, 8 review documents, 8 independent experiments, complete codebase

---

## Executive Summary

The **RSFI (Riemannian System Fidelity Index)** project represents a geometric method for detecting malicious prompts in LLM systems. After comprehensive independent auditing across multiple rounds, the project has revealed both genuine technical merit and significant methodological issues requiring correction before academic submission.

### Key Findings

**Mathematical Core:** ✅ **Sound and Correct**  
- 250 lines of clean implementation
- 510 lines of comprehensive tests (all passing)
- Covers degenerate cases (antipodal points, singular matrices, float32 drift)

**Real Performance:** ⚠️ **0.82-0.86 AUC (honest evaluation)**
- Best result: **0.856 ± 0.009** on Qwen3-Embedding-8B (4096d)
- Consistent +4.0...+5.9 pp gain over naive cosine baseline
- Underperforms LogReg by 2.5-5.3 pp when N_ref ≥ 10

**Critical Issues Identified:**
1. Data leakage in benchmarks (inflates metrics by ~9 pp)
2. Fabricated datasets (9 unique texts × 400 duplicates)
3. Contradictory documentation (AUC 1.0 vs 0.88 vs 0.86)
4. Missing critical experiments (homogeneous datasets, external baselines)

---

## Project Evolution: Who Did What

### First AI (commits until Feb 2024)
- Created mathematical core (geometry, whitening, filter)
- Generated documentation with inflated metrics
- Ran benchmarks with methodological errors

### Second AI - Independent Audit (Kimi, Aug 4, 2026)
**Initial findings (Round 1):**
- Weak configuration test → AUC 0.608
- Discovered fabricated datasets
- Identified data leakage
- Found prior art for all components

### Authors' Rebuttal → Rounds 2-4
**Authors argued:** "Kimi used weak config (MiniLM + raw QR), our method uses mpnet + SVD"

**Re-verification (honest_eval v2-v8):**
- ✅ Authors correct: SVD subspace works (0.82-0.86 AUC)
- ❌ But their protocol inflates by ~9 pp due to leakage
- ✅ 4096d test achieved best honest result: **0.856**

---

## Current Status Summary

### ✅ What Works

1. **Core Method (768d-4096d embedders):**
   - Stable +4-6 pp gain over cosine baseline
   - SVD threat subspace is the key component
   - ZCA-whitening beneficial only at 4096d (+1.9 pp)

2. **Few-Shot Niche:**
   - Competitive with LogReg when N_ref ≤ 5
   - Useful as fast first-stage filter

3. **Code Quality:**
   - Clean, readable implementation
   - Comprehensive test coverage
   - Proper handling of edge cases

### ❌ What's Broken

1. **Benchmarks (0/7 are fully honest):**
   - 3 with data leakage / hyperparameter tuning on test set
   - 2 use synthetic templates (10 texts × duplicates)
   - 1 classifies template wrappers instead of content
   - 1 blocks 0/2000 at defaults (recall=0)

2. **Documentation:**
   - ROC-AUC: 1.0 (README) vs 0.878 (report) vs 0.856 (honest)
   - Latency: 21 μs (README) vs 6.4 ms vs 22.1 μs
   - "Zero-shot" claim (requires 50-200 labeled examples)
   - Threshold contradiction: τ*=0.65 vs τ*=0.0

3. **Missing Experiments:**
   - No homogeneous dataset tests (ToxicChat, XSTest)
   - No head-to-head external baseline comparison
   - No TPR@FPR≤1% metric (critical for guardrails)
   - No statistical tests (DeLong, bootstrap CI)

---

## Honest Performance Summary

### Multi-Model Results (Honest Protocol)

| Embedder | Dim | RSFI (clean) | Naive Cosine | LogReg (ref) |
|----------|-----|--------------|--------------|--------------|
| MiniLM-L6 | 384d | 0.761 ± 0.021 | 0.770 | 0.870 |
| mpnet-base | 768d | 0.821 ± 0.013 | 0.785 | 0.874 |
| bge-base | 768d | 0.822 ± 0.007 | 0.770 | 0.870 |
| **Qwen3-8B** | **4096d** | **0.856 ± 0.009** | 0.797 | 0.881 |

**Key Facts:**
- SVD subspace: +4.0...+5.6 pp (only working component)
- ZCA-whitening: negative on 384-768d, positive only on 4096d
- Model-dependent stability: 0.709-0.822 across embedders
- LogReg consistently better at equal data budget

---

## Action Plan: Path to Honest Publication

### Priority 0: Critical Experiments (required)

**E2: Homogeneous Datasets (3-5 days)**
```python
# ToxicChat-0124 (LMSYS) - real toxic user prompts
# XSTest-v2 - contrastive pairs ("kill process" vs "kill person")
# Expected: all methods degrade, question is whether RSFI degrades LESS
```

**E3: Operating Point (1-2 days)**
```python
# TPR at FPR ≤ 1% / ≤ 0.1% (more important than AUC for guardrails)
# PR-AUC (precision-recall, GradSafe standard)
# Threshold calibrated on validation only
```

**E7: External Baselines (5-7 days)**
```python
# Head-to-head on SAME data:
# - Meta Prompt-Guard-86M
# - ProtectAI deberta-v3-base
# - ITMO codebook k-NN
# - LogReg, naive cosine
```

**Documentation Fixes (1 day)**
- Remove AUC 1.0 → replace with 0.856
- Remove "zero-shot" → "few-shot (50-200 examples)"
- Remove "22 μs" → "~10-15 ms with embedding"
- Add "Limitations" section

**Code Fixes (1 day)**
- Remove data leakage from all benchmarks
- Move fabricated datasets to `legacy/`
- Create `honest_eval_final.py` - reproducible script

### Priority 1: Strengthening Experiments

- E5: Whitening stability at 4096d (sweep N_calib)
- E6: Adaptive attacks (GCG, obfuscation) - boundary of applicability
- E10: Statistical tests (DeLong, bootstrap CI, 10 seeds)

---

## Realistic Publication Assessment

### Current State: **"Working Average Method with Inflated Documentation"**

| Criterion | Status |
|-----------|--------|
| Mathematics | ✅ Correct |
| Core Code | ✅ Clean, tested |
| Real Performance | ⚠️ 0.82-0.86 (not 1.0) |
| Novelty | ❌ Absent (prior art) |
| Benchmarks | ❌ 0/7 honest |
| Documentation | ❌ Contradictory |
| **Current state** | ❌ Desk reject |

### After P0 Package: **"Honest Study of Applicability Boundaries"**

**Realistic Venues:**
- ✅ Workshop (NeurIPS SoLaR, ACL SRW, ICLR TinyPapers)
- ✅ University proceedings (tier 3)
- ✅ Tech report (arXiv)
- ⚠️ Q1 / High-tier journal - **unlikely** (no novelty, loses to LogReg)

---

## Recommended Strategy

### Honest Formulation for Paper

**Abstract:**
> We investigate geometric filtering for jailbreak detection based on ZCA-whitening and SVD threat subspaces. On heterogeneous datasets, the method achieves ROC-AUC 0.82-0.86 on embedders ≥768d, consistently outperforming naive cosine by +4-6 pp. However, it underperforms logistic regression trained on the same reference set by 2.5-5.3 pp. We identify a few-shot niche (N_ref ≤ 5) and dimension-dependent ZCA behavior (beneficial at 4096d, detrimental at 384-768d). The method is suitable as a fast first-stage filter in multi-layer defense.

### Defense Positioning

**Emphasize:**
- ✅ Solid mathematical core (510 lines of tests, all pass)
- ✅ Reproducible experiments (honest_eval v2-v8)
- ✅ Best honest result 0.856 on SOTA embedder
- ✅ Stable gain on 768-4096d models

**Acknowledge:**
- ⚠️ Limited novelty (combination of known methods)
- ⚠️ Loses to LogReg at N_ref ≥ 10
- ⚠️ Untested on homogeneous distributions (main risk)
- ⚠️ Dataset stylistically heterogeneous (may inflate)

---

## Time Estimates

### Minimum Path (2-3 weeks)
- E2 (ToxicChat/XSTest): 3-5 days
- E3 (TPR@FPR): 1-2 days
- E7 (external baselines): 5-7 days
- Code + doc fixes: 2 days
- **Total: 11-16 days active work**

### Enhanced Path (1-2 months)
- Everything from minimum
- P1 experiments (E5, E6, E10): 7-10 days
- Paper writing: 5-7 days
- **Total: 23-33 days active work**

---

## Main Conclusion

**The method works, but not at the level claimed in current documentation.**

The task is **not to remake the method** (it's already good for its niche), but **to bring experiments and documentation to honest state**.

This is achievable in 2-4 weeks and will result in defendable thesis and publishable work (workshop / tier-3 level).

**Key Insight:** Negative results, honestly measured and explained, are more valuable than inflated SOTA claims. The scientific community respects honesty.

---

**All materials for continuation:**
- `AUDIT_SUMMARY.md` - full current state analysis
- `REALISTIC_ROADMAP.md` - step-by-step plan with code
- `EXPERIMENT_PLAN.md` - original authors' plan (P0-P2)
- `FINAL_SUMMARY_RU.md` - Russian version for local team
- `honest_eval_v2.py` ... `v8.py` - working honest experiments

**Next Step:** Read `REALISTIC_ROADMAP.md` and start with E2 (homogeneous datasets) - the critical experiment that determines everything else.
