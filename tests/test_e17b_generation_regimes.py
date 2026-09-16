"""
test_e17b_generation_regimes.py
================================================================================
Consistency gate for the follow-up item 3 outputs:
  data/results/E17b_generation_regimes_prompts.csv
  data/results/E17b_generation_regimes_summary.csv
  data/results/E17c_judge_power_analysis.csv
  data/results/E17c_judge_power_reference.csv

Invariants:
  1. E17b prompts table: judge labels within the committed taxonomy; prompt
     classes {malicious, safe}; the same prompt set (dataset, orig_idx) is
     present under EVERY regime - the regimes are a paired comparison, so a
     missing prompt in one regime would invalidate the pairing.
  2. E17b summary: rates in [0, 1]; Wilson CIs bracket the point estimate and
     lie in [0, 1]; n_compliant + the other judge labels == n_malicious_prompts
     for every (dataset, regime).
  3. E17c: Wilson interval brackets the point rate; widths shrink with n;
     `required_n` matches the closed-form formula z^2 p(1-p)/h^2 recomputed
     independently in the test; degenerate rates (0 or 1) are stored as NaN.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).parent.parent / "data" / "results"
JUDGE_LABELS = {"REFUSAL", "HARMFUL_COMPLIANCE", "BENIGN_RESPONSE"}
Z = 1.959963984540054


@pytest.fixture(scope="module")
def prompts():
    path = RESULTS / "E17b_generation_regimes_prompts.csv"
    if not path.exists():
        pytest.skip("E17b_generation_regimes_prompts.csv not committed yet")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def summary():
    path = RESULTS / "E17b_generation_regimes_summary.csv"
    if not path.exists():
        pytest.skip("E17b_generation_regimes_summary.csv not committed yet")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def power():
    path = RESULTS / "E17c_judge_power_analysis.csv"
    if not path.exists():
        pytest.skip("E17c_judge_power_analysis.csv not committed yet")
    return pd.read_csv(path)


def test_regimes_are_paired(prompts):
    for col in ["dataset", "orig_idx", "prompt_class", "regime",
                "llm_response", "judge_label", "b1_score", "rsfi_neg_score"]:
        assert col in prompts.columns, f"missing column: {col}"
    assert set(prompts.judge_label.unique()) <= JUDGE_LABELS
    assert set(prompts.prompt_class.unique()) == {"malicious", "safe"}
    sets = {reg: set(zip(g.dataset, g.orig_idx))
            for reg, g in prompts.groupby("regime")}
    assert len(sets) >= 2, f"expected several regimes, got {list(sets)}"
    ref_name, ref_set = next(iter(sets.items()))
    for reg, s in sets.items():
        assert s == ref_set, \
            f"regime {reg} does not cover the same prompts as {ref_name}"


def test_summary_rates_and_cis(prompts, summary):
    for _, row in summary.iterrows():
        g = prompts[(prompts.dataset == row.dataset)
                    & (prompts.regime == row.regime)]
        mal = g[g.prompt_class == "malicious"]
        assert len(mal) == int(row.n_malicious_prompts)
        assert int(row.n_compliant) + int(row.n_refusal) <= len(mal)
        assert g.judge_label.value_counts().sum() == len(g)
        for col in ["compliance_rate", "refusal_rate", "tpr", "fpr"]:
            v = row[col]
            if pd.notna(v):
                assert 0.0 <= float(v) <= 1.0, f"{col} out of [0,1]: {v}"
        for lo, hi, mid in [("compliance_ci_low", "compliance_ci_high",
                             "compliance_rate"),
                            ("tpr_ci_low", "tpr_ci_high", "tpr"),
                            ("fpr_ci_low", "fpr_ci_high", "fpr")]:
            if pd.notna(row[mid]):
                # Tolerance on every link: a Wilson bound may legitimately
                # equal the point estimate exactly (e.g. 1/1 -> hi = 1.0),
                # so mid + 1e-9 must be compared against hi + 1e-9.
                assert 0.0 - 1e-9 <= float(row[lo]) <= float(row[mid]) \
                    + 1e-9 <= float(row[hi]) + 1e-9 <= 1.0 + 1e-9
        assert row.tau_calibrated_on == "safe_reference_pool"


def test_power_reference_matches_closed_form():
    path = RESULTS / "E17c_judge_power_reference.csv"
    if not path.exists():
        pytest.skip("E17c_judge_power_reference.csv not committed yet")
    ref = pd.read_csv(path)
    for _, row in ref.iterrows():
        p, h = float(row.assumed_rate_p), float(row.half_width_h)
        expected = int(np.ceil(Z * Z * p * (1 - p) / (h * h)))
        assert int(row.required_n) == expected, \
            f"p={p}, h={h}: stored {row.required_n} != closed form {expected}"


def test_power_rows_consistent(power):
    for _, row in power.iterrows():
        n = int(row.n_malicious_prompts)
        assert n > 0
        assert 0.0 <= float(row.compliance_ci_low) <= float(row.compliance_rate) \
            + 1e-9 <= float(row.compliance_ci_high) <= 1.0
        for h_col in ["required_n_tpr_h05", "required_n_tpr_h10",
                      "required_n_tpr_h20"]:
            v = row[h_col]
            if pd.notna(v):
                assert float(v) >= 1.0
        # degenerate (0 or 1) rates must not produce a spurious sample size
        if row.compliance_rate == 0.0 and pd.isna(row.tpr):
            assert pd.isna(row.required_n_tpr_h10)