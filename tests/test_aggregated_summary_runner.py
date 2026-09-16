"""Run the aggregator, not just checks of previously generated artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import generate_aggregated_summary as agg

RESULTS = Path(__file__).parent.parent / "data" / "results"


def test_main_recomputes_transfer_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "load", lambda name: pd.read_csv(RESULTS / name))
    monkeypatch.setattr(agg, "RESULTS", tmp_path)
    agg.main()
    ci = pd.read_csv(tmp_path / "AGGREGATED_mean_auc_ci.csv")
    assert not ci.duplicated().any()
    raw = pd.read_csv(RESULTS / "E13b_transfer_to_bigbench.csv")
    scores = raw[raw.method != "DELONG_B1w_vs_A1"].copy()
    scores["dataset"] = scores.train_ds + "->" + scores.target_ds
    expected = scores.groupby(["dataset", "embedder", "method"]).roc_auc.agg(
        ["mean", "std", "count"])
    actual = ci[ci.source == "E13b_transfer_BB"].set_index(
        ["dataset", "embedder", "method"]).sort_index()
    assert len(actual) == len(expected) == 320
    np.testing.assert_allclose(actual.mean_auc, expected["mean"])
    np.testing.assert_allclose(actual.std_auc, expected["std"], atol=1e-15)
    np.testing.assert_array_equal(actual.n_seeds, expected["count"])
    assert len(ci[ci.source == "E13_crossdomain"]) == 4
    dl = pd.read_csv(tmp_path / "AGGREGATED_delong.csv")
    assert len(dl[dl.source == "E13b"]) == 80
    assert dl.auc_diff_mean.notna().all()


@pytest.mark.parametrize("pvalues", [[0.01, 0.2, np.nan], [np.nan] * 3])
def test_transfer_delong_direction_and_undefined_pvalues(pvalues):
    rows = []
    for seed, (a1, b1w, p) in enumerate(zip([0.8, 0.7, 1.0],
                                           [0.6, 0.9, 1.0], pvalues)):
        for method, value in [("A1_raw", a1), ("B1w_SigmaW", b1w),
                              ("DELONG_B1w_vs_A1", p)]:
            rows.append(dict(train_ds="train", target_ds="target", embedder="emb",
                             seed=seed, method=method, roc_auc=value))
    result = agg.aggregate_delong_pvalue(pd.DataFrame(rows), "E13b", "B1w_vs_A1").iloc[0]
    assert result.wins_m1 == 1
    assert result.losses == 1
    assert result.ties == 1
    assert result.auc_diff_mean == pytest.approx(0, abs=1e-15)
    valid = np.asarray(pvalues)[np.isfinite(pvalues)]
    assert result.n_valid_p == len(valid)
    if len(valid):
        assert result.p_mean == pytest.approx(valid.mean())
        assert result.frac_p_below_0_05 == pytest.approx((valid < 0.05).mean())
    else:
        assert pd.isna(result.p_mean)
        assert pd.isna(result.p_max)
        assert pd.isna(result.frac_p_below_0_05)
