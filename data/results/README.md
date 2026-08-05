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
- `E6_*.csv` - Adaptive attacks
- `E7_*.csv` - Head-to-head baselines
- `E10_*.csv` - Statistical significance

All new experiments follow honest evaluation:
1. No test set leakage
2. Proper train/val/test splits
3. Real datasets only
4. Statistical significance testing
5. Confidence intervals reported
