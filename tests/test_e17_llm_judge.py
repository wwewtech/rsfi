"""
test_e17_llm_judge.py - gate for E17 LLM-as-a-Judge ground truth.

Behavioural invariants (ground truth = judge labels):
  - the judge classes come from the committed
    src/analysis/evaluate_results_with_llm_judge.py (import, not copy);
  - prompts are sampled ONLY from the leakage-free E11 test split;
  - blocking flags recomputed from the stored scores and taus must match
    the stored flags (internal consistency);
  - summary TPR/FPR values must be reproducible from the per-prompt rows.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "data" / "results"


def _import(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(mod_name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data():
    prompts_path = RESULTS / "E17_llm_judge_prompts.csv"
    summary_path = RESULTS / "E17_llm_judge_summary.csv"
    if not prompts_path.exists():
        pytest.skip("E17_llm_judge_prompts.csv not committed yet")
    return pd.read_csv(prompts_path), pd.read_csv(summary_path)


def test_judge_source_is_committed():
    """The LLMSafetyJudge class must exist in the committed judge module."""
    judge_mod = _import("judge_src", "src/analysis/evaluate_results_with_llm_judge.py")
    assert hasattr(judge_mod, "LLMSafetyJudge")


def test_prompts_schema_and_labels(data):
    df, _ = data
    for col in ["dataset", "prompt_class", "user_prompt", "llm_response",
                "judge_label", "b1_score", "rsfi_neg_score",
                "blocked_b1", "blocked_rsfi"]:
        assert col in df.columns, f"missing column {col}"
    assert set(df.judge_label.unique()) <= {"REFUSAL", "HARMFUL_COMPLIANCE",
                                            "BENIGN_RESPONSE"}
    assert set(df.dataset.unique()) == {"ToxicChat", "Wild", "XSTest"}
    assert set(df.prompt_class.unique()) == {"malicious", "safe"}
    assert df.user_prompt.str.len().min() > 0
    assert df.llm_response.str.len().min() > 0


def test_blocked_flags_consistent_with_scores(data):
    df, _ = data
    for (d_name), g in df.groupby("dataset"):
        for score_col, tau_col, flag_col in [
                ("b1_score", "tau_b1", "blocked_b1"),
                ("rsfi_neg_score", "tau_rsfi_neg", "blocked_rsfi")]:
            assert tau_col in g.columns, f"missing {tau_col}"
            recomputed = g[score_col] >= g[tau_col]
            assert (recomputed == g[flag_col]).all(), \
                f"{d_name}/{flag_col}: stored flags disagree with score>=tau"


def test_summary_reproducible_from_prompts(data):
    df, summ = data
    for _, row in summ.iterrows():
        g = df[df.dataset == row.dataset]
        harmful, benign = g[g.true_is_harmful], g[~g.true_is_harmful]
        col = "blocked_b1" if row.method == "B1" else "blocked_rsfi"
        assert len(harmful) == row.n_harmful
        assert len(benign) == row.n_benign
        if len(harmful):
            assert abs(float(harmful[col].mean())
                       - float(row.tpr_harmful_blocked)) < 1e-6
        if len(benign):
            assert abs(float(benign[col].mean())
                       - float(row.fpr_benign_blocked)) < 1e-6


def test_min_sample_size(data):
    df, _ = data
    # 3 datasets x 2 classes x >=20 prompts = >=120 rows (default is 40/class)
    assert len(df) >= 120, f"too few judged prompts: {len(df)}"
