# Legacy Experiments Archive

This directory contains files kept **for provenance only**. They are superseded by the
final verified pipeline (`experiments/E2*`–`E11`, `src/rsfi`) and must **not** be cited.

## `experiments/archive/` (code)

Early-stage scripts predating the leakage-free protocol:

- `honest_eval*.py`, `test1..test4*.py`, `test_ai.py` — pre-audit era with test-set
  leakage (ZCA calibrated on test data) and synthetic data (9 texts x 400 copies);
  their claims ("0% false bans", "AUC 1.0") were retracted by later audits.
- `honest_eval_final.py`, `run_*_benchmark.py`, `sota_benchmark_suite.py`,
  `*_discovery.py`, `adaptive_lid_vs_baseline.py` — interim exploration scripts
  (moved from `experiments/` root during the cleanup); their outputs
  (`data/results/real_llm_sfi_*.csv`, `data/results/sfi_wild_10k_results.csv`,
  `data/figures/fig1..fig4_*.png`) are retained as committed history but are not
  part of the verified result chain either.

## `docs/archive/figures_legacy/`

Superseded figure duplicates (`*_2.png`, 03.08.2026 rerun) for which no generator
script is committed. The `data/figures/*.png` set has the same status: no committed
generator script references them.

## Deleted during cleanup (recoverable from git history)

- `data/reports/monte_carlo_10h_results.csv`,
  `data/reports/monte_carlo_summary_for_paper.csv`,
  `data/reports/scaling_laws_results.csv` — outputs of the deleted
  `experiments/archive/honest_eval_v7/v8/parse_monte_carlo.py` era.
- `data/results/E5_whitening_stability.csv` — empty (2 bytes), referenced nowhere.
- `data/results/enhanced_benchmark_results.csv`,
  `data/results/optimal_geometric_results.csv` — untracked legacy outputs of the
  interim scripts above, referenced nowhere.

The verified, citable result chain lives in `data/results/E2d*`–`E11_*.csv`
(see `data/results/README.md`) and is covered by the consistency test suite
(`tests/test_report_consistency.py`, 210 tests total).
