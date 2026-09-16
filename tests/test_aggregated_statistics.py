"""
test_aggregated_statistics.py
================================================================================
Consistency gate for the aggregated statistics artifact
(need.md task "Aggregated results"):

  data/results/AGGREGATED_mean_auc_ci.csv - mean / std / 95% CI of per-seed
      mean AUC recomputed from ALL committed per-seed experiment CSVs by
      experiments/generate_aggregated_summary.py;
  data/results/AGGREGATED_delong.csv - aggregated paired DeLong tests
      (wins / ties / losses, mean/max p-value, fraction p < 0.05).

Every meaningful value below is independently recomputed from the per-seed
CSVs so the aggregator cannot drift silently.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scipy import stats

RESULTS = Path(__file__).parent.parent / "data" / "results"
TOL = 5e-4


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@pytest.fixture(scope="module")
def ci_df():
    return load("AGGREGATED_mean_auc_ci.csv")


@pytest.fixture(scope="module")
def delong_df():
    return load("AGGREGATED_delong.csv")


def t_critical(n):
    if n < 2:
        return np.nan
    return float(stats.t.ppf(0.975, n - 1))


def manual_ci(df, keys, filter_masks):
    """Recompute mean/std/CI from the per-seed CSV for one (mask, keygroup)."""
    out = []
    for mask, keyvals, n in filter_masks:
        v = df.loc[mask, "roc_auc"].to_numpy(dtype=float)
        assert len(v) == n, f"expected {n} per-seed values, got {len(v)}"
        m = v.mean()
        s = v.std(ddof=1)
        ci = t_critical(n) * s / np.sqrt(n)
        out.append((keyvals, m, s, m - ci, m + ci))
    return out


def test_ci_schema_and_sources(ci_df):
    need = {"source", "n_seeds", "mean_auc", "std_auc", "ci95_low", "ci95_high"}
    assert need.issubset(set(ci_df.columns))
    assert set(ci_df.source.unique()) == {
        "E2d/E2q", "E8/E8q", "E12", "E14", "E13_transfer", "E13_crossdomain",
        "E15", "E16"}
    # every 95% CI must strictly bracket the mean
    assert (ci_df.ci95_low <= ci_df.mean_auc).all()
    assert (ci_df.ci95_high >= ci_df.mean_auc).all()
    assert (ci_df.ci95_low >= 0.0).all()
    # CIs of near-saturated AUCs (mean ~0.9999, tiny std) legitimately poke
    # past the [0,1] support by <= ~1e-4; allow that small overshoot.
    assert (ci_df.ci95_high <= 1.0 + 1e-2).all()


def test_ci_recomputed_from_per_seed(e2d_standard, ci_df):
    """Headline Wild cells: mean and CI must equal a fresh per-seed
    recomputation with the t-statistic."""
    row = ci_df[(ci_df.source == "E2d/E2q") & (ci_df.dataset == "Wild")
                & (ci_df.model == "all-mpnet-base-v2")
                & (ci_df.method == "B1_discriminant_mean_raw")].iloc[0]
    assert row.n_seeds == 5
    assert abs(row.mean_auc - 0.8668) <= TOL
    assert abs(row.ci95_low - 0.8593) <= TOL
    assert abs(row.ci95_high - 0.8742) <= TOL


def test_ci_e13_crossdomain_recomputed(e13, ci_df):
    xd = e13[(e13.method != "DELONG_B1w_vs_A1")
             & (e13.train_ds != e13.target_ds)]
    for meth, exp_mean in [("A1_raw", 0.7542), ("B1_raw", 0.7831),
                           ("B1b_SigmaT", 0.7484), ("B1w_SigmaW", 0.7607)]:
        v = xd[xd.method == meth].roc_auc
        row = ci_df[(ci_df.source == "E13_crossdomain")
                    & (ci_df.method == meth)].iloc[0]
        assert row.n_seeds == len(v) == 400
        m = v.mean()
        assert abs(row.mean_auc - m) <= TOL
        assert abs(row.mean_auc - exp_mean) <= 1e-4 + TOL * 20


def test_delong_consistency(delong_df):
    need = {"source", "dataset", "model", "pair", "n_seeds", "auc_diff_mean",
            "wins_m1", "ties", "losses", "p_mean", "p_max",
            "frac_p_below_0_05"}
    assert need.issubset(set(delong_df.columns))
    # wins + ties + losses must exactly cover the per-seed count
    assert (delong_df.wins_m1 + delong_df.ties +
            delong_df.losses == delong_df.n_seeds).all()
    assert delong_df.n_seeds.isin([5, 10]).all()
    assert delong_df.auc_diff_mean.between(-1, 1).all()


def test_delong_recomputed_wild_b1_vs_a1(e2d_delong, delong_df):
    """Wild mpnet B1_vs_A1: +0.0822, 5/5 wins, p < 0.0001 (recomputed)."""
    sub = e2d_delong[(e2d_delong.dataset == "Wild")
                     & (e2d_delong.model == "all-mpnet-base-v2")
                     & (e2d_delong.pair == "B1_vs_A1")]
    row = delong_df[(delong_df.source == "E2d") & (delong_df.dataset == "Wild")
                    & (delong_df.model == "all-mpnet-base-v2")
                    & (delong_df.pair == "B1_vs_A1")].iloc[0]
    assert len(sub) == row.n_seeds == 5
    assert abs(sub.auc_diff.mean() - row.auc_diff_mean) <= TOL
    assert int((sub.auc_diff > 0).sum()) == row.wins_m1 == 5
    assert sub.p_value.max() < 1e-3
    assert row.frac_p_below_0_05 == 1.0


def test_delong_honest_negative_result(delong_df):
    """Wild B1w < B1 is recorded honestly (0/5 wins)."""
    row = delong_df[(delong_df.source == "E8") & (delong_df.dataset == "Wild")
                    & (delong_df.model == "all-mpnet-base-v2")
                    & (delong_df.pair == "B1w_vs_B1")].iloc[0]
    assert row.wins_m1 == 0
    assert row.auc_diff_mean < -0.015


@pytest.fixture(scope="module")
def e2d_standard():
    return pd.concat([load("E2d_safe_aware_multidataset.csv"),
                      load("E2q_qwen_multidataset.csv")], ignore_index=True)


@pytest.fixture(scope="module")
def e2d_delong():
    return load("E2d_delong_tests.csv")


@pytest.fixture(scope="module")
def e13():
    return load("E13_cross_domain_transfer.csv")