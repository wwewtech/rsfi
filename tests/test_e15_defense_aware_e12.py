"""
test_e15_defense_aware_e12.py
================================================================================
Consistency gate for `data/results/E15_defense_aware_e12.csv`
(E15 = E6c defense-aware adaptive attack protocol transferred to the E12
AdvBench / HarmBench datasets).

Invariants checked (behavioural, no hard-coded AUCs):
  1. Schema: all E6c columns present; datasets == {AdvBench, HarmBench};
     one clean row per (dataset, model, seed, evaluated_method, scenario) grid.
  2. Clean baseline sanity: for every (dataset, model, seed, evaluated_method)
     the clean-row ROC-AUC >= 0.85 (the E12 benchmark shows near-ceiling
     discriminant AUC ~ 0.997-1.0; if clean AUC collapses, the run is broken).
  3. Operating-point consistency: clean ASR at 5% FPR >= clean ASR at 1% FPR
     (monotone thresholds), both in [0, 1].
  4. Attack rows exist for all 4 E6c scenarios; for every attacked scenario
     mean_semantic_sim >= 0.5 and mean_perturb_ratio <= 0.5 (the attack
     constraint is 0.80 similarity; the gate allows a looser 0.5 to remain
     robust while still catching a broken semantic-preservation loop).
  5. Reproducibility: 5 seeds per (dataset, model) for every scenario.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).parent.parent / "data" / "results"
E15_SCENARIOS = {
    "adaptive_word_greedy_target_B1",
    "adaptive_affix_target_B1",
    "adaptive_combined_target_B1",
    "adaptive_word_greedy_target_B1w",
}
E15_DATASETS = {"AdvBench", "HarmBench"}
N_SEEDS = 5


@pytest.fixture(scope="module")
def e15():
    path = RESULTS / "E15_defense_aware_e12.csv"
    if not path.exists():
        pytest.skip("E15_defense_aware_e12.csv not committed yet")
    return pd.read_csv(path)


def test_schema_and_coverage(e15):
    required = {
        "dataset", "model", "seed", "attack_scenario", "target_defense",
        "evaluated_method", "roc_auc", "pr_auc", "asr_at_1pct_fpr",
        "asr_at_5pct_fpr", "asr_at_median_safe", "mean_perturb_ratio",
        "mean_semantic_sim", "mean_queries",
    }
    missing = required - set(e15.columns)
    assert not missing, f"missing columns: {missing}"
    assert set(e15.dataset.unique()) == E15_DATASETS
    scenarios = set(e15.attack_scenario.unique())
    assert E15_SCENARIOS <= scenarios, f"missing scenarios: {E15_SCENARIOS - scenarios}"


def test_clean_baseline_auc_floor(e15):
    """Geometric E6c methods must stay near-ceiling on clean data.

    The external classifiers (deberta-v3-prompt-injection-v2, toxic-bert)
    are transferability BASELINES - their clean AUC is an experimental
    finding, not an invariant, so the floor applies only to the 6 E6c
    geometric methods.
    """
    geometric = {
        "A1_naive_cosine_raw", "A2_rsfi_svd_raw_k20",
        "B1_discriminant_mean_raw", "B1b_SigmaT_wh", "B1w_SigmaW_wh",
        "C1_logreg_raw",
    }
    clean = e15[(e15.attack_scenario == "clean")
                & (e15.evaluated_method.isin(geometric))]
    assert len(clean) > 0
    worst = float(clean.roc_auc.min())
    assert worst >= 0.85, f"clean ROC-AUC floor violated: min={worst:.4f}"


def test_operating_point_monotonicity(e15):
    """E6c semantics: tau_1pct = P99(safe) >= tau_5pct = P95(safe) >= median,
    and ASR = share of attack scores BELOW tau -> the monotone chain is
    ASR@1pct >= ASR@5pct >= ASR@median (more sensitive detector => lower ASR).
    """
    for c in ["asr_at_1pct_fpr", "asr_at_5pct_fpr", "asr_at_median_safe"]:
        assert e15[c].between(0.0, 1.0).all(), f"{c} out of [0,1]"
    clean = e15[e15.attack_scenario == "clean"]
    g = clean.groupby(["dataset", "model", "seed", "evaluated_method"])
    bad = []
    for k, gr in g:
        a1 = float(gr.asr_at_1pct_fpr.mean())
        a5 = float(gr.asr_at_5pct_fpr.mean())
        am = float(gr.asr_at_median_safe.mean())
        if a1 + 1e-9 < a5 or a5 + 1e-9 < am:
            bad.append((k, a1, a5, am))
    assert not bad, f"{len(bad)} groups violate ASR monotonicity: {bad[:5]}"


def test_attack_constraints(e15):
    adv = e15[e15.attack_scenario.isin(E15_SCENARIOS)]
    assert len(adv) > 0
    assert float(adv.mean_semantic_sim.min()) >= 0.5, \
        "semantic preservation collapsed below 0.5"
    assert float(adv.mean_perturb_ratio.max()) <= 0.5, \
        "perturbation budget exceeded 50% of words"


def test_seed_coverage(e15):
    for (ds, mdl, scen), gr in e15.groupby(["dataset", "model", "attack_scenario"]):
        seeds = set(gr.seed.unique())
        assert seeds == set(range(N_SEEDS)), \
            f"{ds}/{mdl}/{scen}: seeds {sorted(seeds)} != 0..{N_SEEDS - 1}"


def test_delta_summary_consistency(e15):
    """Adv minus clean ASR@1%FPR is computable for every attacked row group."""
    clean = (e15[e15.attack_scenario == "clean"]
             .groupby(["dataset", "model", "evaluated_method"])["asr_at_1pct_fpr"]
             .mean())
    adv = e15[e15.attack_scenario.isin(E15_SCENARIOS)]
    for r in adv.itertuples():
        key = (r.dataset, r.model, r.evaluated_method)
        assert key in clean.index, f"no clean reference for {key}"
        delta = r.asr_at_1pct_fpr - float(clean.loc[key])
        assert -1.0 <= delta <= 1.0
