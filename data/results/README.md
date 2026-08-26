# Results Directory

This directory contains results from honest, reproducible experiments.

## Active Results

These results follow the honest evaluation protocol (no test leakage, proper splits):

- `sfi_wild_10k_results.csv` - Real-world WildChat jailbreak dataset (10k samples)
- `real_llm_sfi_*.csv` - Real LLM evaluation results with honest calibration

## Legacy Results (./legacy/)

Moved to legacy/ due to methodological issues:
- Test set leakage (ZCA calibrated on test data)
- Fake/synthetic datasets without real attacks
- Inflated metrics from improper evaluation

These are kept for historical reference but should NOT be cited.

## New Experiment Results

Results from E1-E10 experiments (REALISTIC_ROADMAP.md) will appear here:
- `E1_*.csv` - Dataset size scaling
- `E2_*.csv` - Embedding model comparison
- `E3_*.csv` - Operating point analysis
- `E5_*.csv` - Whitening stability
- `E6_adaptive_attacks.csv` - DEPRECATED legacy artifact: degenerate sizes (n_ref=5, n_test_attack=5), single embedder, one-class methods only. Superseded by `E6b_obfuscation_boundary.csv`; kept for audit history only, do NOT cite.
- `E6b_obfuscation_boundary.csv` - Obfuscation boundary for the central Safe-Aware methods (B1/B1b/B1w) plus A1/A2 anchors and C1 ceiling on identical E2d/E8/E9 leakage-free splits (5 seeds, 3 embedders, 3 datasets x {clean, base64, leetspeak, rot13, zero_width, homoglyph}); built-in sanity gate verifies that 'clean' rows reproduce committed E8/E2d means within 5e-4.
- `E7_*.csv` - Head-to-head baselines
- `E10_*.csv` - Statistical significance

All new experiments follow honest evaluation:
1. No test set leakage
2. Proper train/val/test splits
3. Real datasets only
4. Statistical significance testing
5. Confidence intervals reported
