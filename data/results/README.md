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
- `E2_*.csv` - Embedding model comparison (E2d: 3 embedders; E2q: Qwen3-Embedding-8B 4096d on all 3 datasets x 5 seeds)
- `E3_*.csv` - Operating point analysis (strict FPR thresholds)
- `E5_*.csv` - Whitening stability
- `E6_adaptive_attacks.csv` - DEPRECATED legacy artifact: degenerate sizes (n_ref=5, n_test_attack=5), single embedder, one-class methods only. Superseded by `E6b_obfuscation_boundary.csv`; kept for audit history only, do NOT cite.
- `E6b_obfuscation_boundary.csv` - Obfuscation boundary for the central Safe-Aware methods (B1/B1b/B1w) plus A1/A2 anchors and C1 ceiling on identical E2d/E8/E9 leakage-free splits (5 seeds, 3 embedders, 3 datasets x {clean, base64, leetspeak, rot13, zero_width, homoglyph, translit}); built-in sanity gate verifies that 'clean' rows reproduce committed E8/E2d means within 5e-4. `translit` is the full-alphabet keyboard layout switch (ЙЦУКЕН<->QWERTY) added 2026-09-16: it is the single most destructive obfuscation tested (mpnet+XSTest: blind cosine AUC ~ 0.0001, polarity flipped).
- `E7_*.csv` - Head-to-head baselines
- `E8_sigma_w.csv`, `E8q_qwen_sigma_w.csv` - Pooled within-class whitening ($\Sigma_W$) vs total whitening ($\Sigma_T$) across 4 embedders x 3 datasets x 5 seeds.
- `E8_knn.csv` - 1-class vs 2-class k-NN semantic codebook baselines across 3 embedders x 3 datasets x 5 seeds.
- `E8_delong_tests.csv`, `E8q_qwen_delong_tests.csv` - Paired DeLong tests (B1 vs A1, B1b vs B1) per seed, standard embedders and Qwen3-8B.
- `E9_external_baselines.csv` - External published classifiers (ProtectAI deberta-v3-prompt-injection-v2 & unitary/toxic-bert) on shared leakage-free splits.
- `E9b_external_obfuscation.csv` - External published classifiers evaluated under 6 test-time obfuscations across all 3 datasets x 5 seeds.
- `E2b_k_sweep_toxicchat.csv`, `E2c_discriminant_diagnosis.csv` - Supporting diagnostics for the Safe-Aware discriminant family (k-sweep, per-class geometry).
- `E11_rsfi_vs_b1.csv`, `E11_rsfi_tangent_space.csv`, `E11_rsfi_alpha_sweep.csv`, `E11_rsfi_operating_point.csv` - RSFI tangent-space validation: rank equivalence with B1, linear-limit alpha sweep, multi-vector SVD variant, operating point.
- `E10_*.csv` - Statistical significance
- `E12_advbench_harmbench_e2d.csv` - 4th-block data extension (E12): full E2d battery (A/B/C) + E8 Sigma_W block on two NEW dataset pairs: AdvBench (520 harmful, Zou 2023) + 400 Alpaca-safe and HarmBench (400, Mazeika 2024) + 400 Alpaca-safe. 4 embedders x 2 datasets x 5 seeds, leakage-free. Key result: on HarmBench raw cosine degrades (A1 = 0.84-0.95) while all discriminant variants saturate to 0.997-1.0 (+15.7 pp for Qwen).
- `E12_advbench_harmbench_delong.csv` - Union of E2d + E8 DeLong pairs for E12.
- `E12_advbench_harmbench_delong_sigma_w.csv` - E8-style pairs only (B1w vs B1b / B1b vs B1 / B1w vs B1).
- `E12_mode_diagnostics.csv` - Per dataset x embedder geometry (cos(mu_mal, mu_safe), intra-class homogeneity).
- `E13_cross_domain_transfer.csv` - MAXIMALLY OBJECTIVE generalization: reference pools drawn ONLY from the train dataset, scored against ALL other datasets (Wild/ToxicChat/XSTest/AdvBench/HarmBench) without re-calibration. 4 embedders x 5 seeds; records roc_auc, pr_auc, tpr@fpr1/5%, and DELONG_B1w_vs_A1 rows for every (train, target) pair. In-domain (train==target) rows EXCLUDE the reference pool from test (leak-free audit 2026-09: seed collision with the eval draw inflated the diagonal AUC to ~0.999); the degenerate XSTest diagonal (all positives fit the ref pool) is skipped.
- `E13_cross_domain_summary.csv` - Per (train_ds, target_ds, embedder, method) mean/std AUC over seeds for E13.
- `E14_operating_point_e12.csv` - Operating-point / calibration audit on AdvBench+HarmBench: TPR@FPR 1/5/10%, FPR@TPR 90%, Brier and ECE-10 (calibrated on reference pools only, no test leakage).
- `E15_defense_aware_e12.csv` - E6c defense-aware adaptive attack protocol transferred verbatim to the E12 data (AdvBench/HarmBench + Alpaca-safe, 3 standard embedders x 5 seeds, 60 attack samples/seed, same scenario set as E6c; Qwen3-8B opt-in via E15_INCLUDE_QWEN=1). Reports ASR@1%FPR per evaluated method (target defenses B1/B1w) plus the delta ASR_adv - ASR_clean (the E12 blocks have no contrastive safe-only regime, so the delta isolates the attack effect from the operating-point effect). Reuses E6c attack engine + E12 dataset builders via import.
- `E16_cross_domain_battery.csv` (+ `_delong.csv`, `_diagnostics.csv`) - Cross-domain expansion of E12: malicious side identical to E12 (AdvBench / HarmBench), safe pools from the OFFICIAL google/BIG-bench repo task.json caches (`data/raw/bigbench/`): BB_Safe = 1200 BigBench inputs (hyperbaton + navigate + temporal_sequences, 400 each, pure puzzle/grammar domain), BB_Mixed = 200 disjoint Alpaca self-contained instructions + 400 BigBench inputs (style-heterogeneous safe class). Full E2d battery + E8 Sigma_W block via E12 `run_battery` reuse, 4 embedders x 5 seeds. Key result: B1b/B1w/C1 ROC-AUC = 1.0000 on both BigBench-only and mixed safe pools for all embedders (incl. Qwen3-8B), while the naive A1 cosine degrades on HarmBench_BB_Mixed (0.7654 mean AUC) - whitened geometric methods are robust to the domain-shifted safe class, the naive prototype is not.
- `E17_llm_judge_prompts.csv`, `E17_llm_judge_summary.csv` - LLM-as-a-Judge ground truth for the E11/E2e line: prompts sampled ONLY from the leakage-free E11 test split (40 per class per dataset), answers generated by Qwen2.5-1.5B-Instruct (greedy; generator != judge), labels from the committed `LLMSafetyJudge` (Qwen2.5-3B-Instruct, src/analysis/evaluate_results_with_llm_judge.py - the startup NameError (os/Path used before import) was fixed 2026-09-16). Summary reports TPR (share of judge-labeled harmful generations blocked) and FPR (benign/refusal blocked) for B1 and negated-E11-RSFI at tau calibrated on the safe reference pool only (1% FPR target).
- `AGGREGATED_mean_auc_ci.csv` - Aggregated 95% confidence intervals of per-seed mean AUC (t-distribution) recomputed from ALL per-seed CSVs (E2d/E2q, E8/E8q, E12, E13, E14); generated by `experiments/generate_aggregated_summary.py`.
- `AGGREGATED_delong.csv` - Aggregated paired DeLong tests: per (source, dataset, model, pair) - wins/ties/losses over seeds, mean/max p-value, fraction of seeds with p<0.05; generated by `experiments/generate_aggregated_summary.py`.

All new experiments follow honest evaluation:
1. No test set leakage
2. Proper train/val/test splits
3. Real datasets only
4. Statistical significance testing
5. Confidence intervals reported
