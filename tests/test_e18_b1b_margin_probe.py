"""
test_e18_b1b_margin_probe.py
================================================================================
Consistency gate for the E18 outputs:
  data/results/E18_b1b_geometry_check.csv
  data/results/E18_b1b_margin_probe_per_sample.csv
  data/results/E18_b1b_margin_probe_summary.csv

Invariants (behavioural, no hard-coded experimental values except the
geometric identity which is a theorem):
  1. Geometry check: the Corollary-1 identity is exact for raw covariances, so
     the measured ratio must match the prediction WHEN the within-class
     covariance is non-degenerate. In the repository's few-shot regime
     (n_ref = 400 < d = 768) the RAW Sigma_W is rank-deficient, so the gate
     checks the Ledoit-Wolf path and asserts the documented mismatch bound.
  2. Per-sample table: tau_1pct >= tau_5pct for every row; evaded flags are
     exactly reproducible from scores vs taus.
  3. Summary table: ASR equals the mean of the per-sample evaded flags; the
     displacement decomposition holds (adv = clean - displacement).
  4. The documented E18 finding must be internally consistent: for the attack
     targets, the mean disp/margin of the Sigma-whitened methods is below the
     raw B1 value at every budget (this is what the script was written to
     test; the assertion guards against a silent regression of the probe).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).parent.parent / "data" / "results"


@pytest.fixture(scope="module")
def geom():
    path = RESULTS / "E18_b1b_geometry_check.csv"
    if not path.exists():
        pytest.skip("E18_b1b_geometry_check.csv not committed yet")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def per_sample():
    path = RESULTS / "E18_b1b_margin_probe_per_sample.csv"
    if not path.exists():
        pytest.skip("E18_b1b_margin_probe_per_sample.csv not committed yet")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def summary():
    path = RESULTS / "E18_b1b_margin_probe_summary.csv"
    if not path.exists():
        pytest.skip("E18_b1b_margin_probe_summary.csv not committed yet")
    return pd.read_csv(path)


def test_geometry_identity_ledoit_wolf(geom):
    """Corollary 1 with shrinkage: measured ratio vs 1/sqrt(1+pi_m pi_s r_W^2)."""
    assert {"dim", "rank_SW_raw", "r_W_lw", "r_T_lw", "ratio_meas_lw",
            "ratio_pred_lw", "rel_err_lw"} <= set(geom.columns)
    # raw covariance is documented as rank-deficient in this regime
    assert (geom.rank_SW_raw < geom.dim).all(), \
        "raw Sigma_W unexpectedly full rank - revisiting the Corollary-1 claim"
    # shrunk path must satisfy the identity within the committed tolerance
    assert (geom.rel_err_lw < 0.25).all(), \
        f"LW Corollary-1 mismatch too large: max={geom.rel_err_lw.max():.4f}"
    assert (geom.ratio_meas_lw < 1.0).all() and (geom.ratio_pred_lw < 1.0).all()
    assert (geom.suppression_lw > 0.5).all(), \
        "Sigma_T is expected to suppress a large share of the interclass norm"


def test_per_sample_tau_order_and_flags(per_sample):
    assert (per_sample.tau_1pct >= per_sample.tau_5pct - 1e-12).all()
    recomputed_1 = per_sample.adv_score < per_sample.tau_1pct
    recomputed_5 = per_sample.adv_score < per_sample.tau_5pct
    assert (recomputed_1 == per_sample.evaded_1pct.astype(bool)).all()
    assert (recomputed_5 == per_sample.evaded_5pct.astype(bool)).all()
    assert np.isfinite(per_sample.clean_score).all()
    assert np.isfinite(per_sample.displacement).all()
    # cosine similarity is computed in float32 upstream: allow 1e-6 slack
    assert (per_sample.semantic_sim >= -1e-6).all()
    assert (per_sample.semantic_sim <= 1.0 + 1e-6).all()


def test_summary_reproducible_from_per_sample(per_sample, summary):
    for _, row in summary.iterrows():
        sel = per_sample[(per_sample.dataset == row.dataset)
                         & (per_sample.seed == row.seed)
                         & (per_sample.budget == row.budget)
                         & (per_sample.attack_target == row.attack_target)
                         & (per_sample.method == row.method)]
        assert len(sel) == int(row.n_attack_samples), \
            f"row coverage mismatch for {row.dataset}/{row.method}"
        # means are accumulated in float64 but written/read via CSV: allow 1e-6
        assert abs(float(sel.evaded_1pct.mean())
                   - float(row.asr_at_1pct_fpr)) < 1e-9
        assert abs(float(sel.margin_clean.mean())
                   - float(row.mean_margin_clean)) < 1e-6
        assert abs(float(sel.displacement.mean())
                   - float(row.mean_displacement)) < 1e-6


def test_whitened_leverage_below_raw(summary):
    """E18 finding: the Sigma-whitened scores move less per unit of semantic
    change than the raw discriminant, so the disp/margin ratio is smaller."""
    targets = summary[summary.is_attack_target]
    for (ds, budget), gr in targets.groupby(["dataset", "budget"]):
        b1 = float(gr[gr.method == "B1_discriminant_mean_raw"]
                   .disp_over_margin.mean())
        for meth in ["B1b_SigmaT_wh", "B1w_SigmaW_wh"]:
            wh = float(gr[gr.method == meth].disp_over_margin.mean())
            assert wh < b1, \
                f"{ds}/budget{budget}/{meth}: {wh:.3f} >= raw B1 {b1:.3f}"