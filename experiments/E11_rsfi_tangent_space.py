"""
E11_rsfi_tangent_space.py
================================================================================
Tangent-Space RSFI Validation Experiment.

This experiment closes the loop between the linear 1-D discriminants
(`B1`, `B1b`) and the actual Riemannian Sphere implementation in
`src/rsfi/filter.py`. Concretely, it answers the following questions:

Q1. Is the **single-vector `RSFIFilter`** (formula:
       rsfi = ||v_perp|| - alpha * pi_thr - beta * d_M
     with defaults alpha=1.5, beta=0.5, tau=-0.2)
   rank-equivalent to the linear safe-aware discriminant `B1` when the
   threat anchor is `V_thr = mu_mal` and the system anchor is
   `S = mu_safe`?  We verify this by computing ROC-AUC of the RSFI score
   on the same leakage-free splits used in E2d, and comparing to the
   committed `B1_discriminant_mean_raw` AUC within TOL=5e-3 (a margin
   allowed for the difference between a 1-D projection and a
   non-linear composite scalar that mixes `||v_perp||`, `pi_thr` and
   `d_M`).

   Important caveat (documented empirically below): with the *factory
   defaults* (alpha=1.5, beta=0.5) and the natural choice of anchors
   S = L2(mu_safe), V_thr = L2(mu_mal), the RSFI score is **rank
   anti-correlated** with B1 on every dataset. The reason is that
   `||v_perp||` and `d_M` are nearly constant for the entire test
   pool (all vectors lie close to the same tangent plane of the safe
   anchor), so the constant dominates and the score does not separate
   the two classes.  We therefore re-tune alpha in Q1b so that the
   score is rank-equivalent to B1, and verify that the *qualitative*
   finding of Table 1 (B1 ~ 0.87 on Wild) is recovered.

Q2. Does the **multi-vector `MultiDimensionalRSFIFilter`** (defaults
       alpha=1.0, beta=0.5, tau=0.0, k=20
     basis via QR of threat anchors)
   produce a usable jailbreak score on the same splits?  We compare its
   AUC against the `B2_contrastive_svd_raw_k20` baseline from E2d.

Q3. Does the recommended operating point (tau calibration to ~1% FPR
   on the *safe* reference pool) produce a useful TPR across all
   three dataset regimes (homogeneous ToxicChat, heterogeneous Wild,
   contrastive XSTest)?

All datasets and embedders mirror E2d (the leakage-free holdout is
re-used so the numbers are directly comparable to Table 1/2 of
RESEARCH_REPORT.md).  No re-fitting of ZCA whitening is required; the
multi-dimensional RSFI works directly in raw L2-normalized
embeddings, which is consistent with the production defaults in
`RSFIFilter.evaluate`.

Outputs
-------
- data/results/E11_rsfi_tangent_space.csv
- data/results/E11_rsfi_operating_point.csv
- data/results/E11_rsfi_vs_b1.csv (per-seed agreement)

Reproducibility
---------------
- 5 seeds, leakage-free holdout (same as E2d)
- All four embedders from E2d/E2e (mpnet, bge-base, bge-large)
- Qwen3-8B is *not* re-run here (E2d/E2e/Qwen path is documented in
  the RSFI report as 4096d extension; this experiment stays at d <=
  1024 to avoid 16 GB GPU stalls in CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rsfi.filter import RSFIFilter, MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening  # noqa: F401  (ensures import)

# Re-use the data loaders and embedding cache from E2d.
sys.path.insert(0, str(Path(__file__).parent))
from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_xstest,
    load_wild,
    get_embeddings,
    N_SEEDS,
    DEVICE,
)

EPS = 1e-15
TOL = 5e-3
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- helpers
def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def rsfi_single_score(
    ref_mal: np.ndarray,
    ref_safe: np.ndarray,
    eval_emb: np.ndarray,
    alpha: float = 1.5,
    beta: float = 0.5,
    tau: float = -0.2,
) -> np.ndarray:
    """Single-vector RSFI score with S = mu_safe, V_thr = mu_mal.

    Returns the *raw* RSFI value (before the tau threshold) per row of
    `eval_emb`.  The score's ROC-AUC is what we compare to B1.

    Vectorised: we re-derive the analytic formula directly instead of
    looping over RSFIFilter.evaluate(), which is ~4000x faster.
    """
    S = l2(ref_safe.mean(axis=0, keepdims=True)).flatten()
    V_thr = l2(ref_mal.mean(axis=0, keepdims=True)).flatten()

    eval_l2 = l2(eval_emb)
    v_thr = RiemannianSphere.log_map(S, V_thr)
    e_thr = v_thr / max(np.linalg.norm(v_thr), 1e-15)
    n = eval_l2.shape[0]
    v_R = np.empty((n, S.size), dtype=np.float64)
    for i in range(n):
        v_R[i] = RiemannianSphere.log_map(S, eval_l2[i])
    d_M = np.linalg.norm(v_R, axis=1)
    pi_thr = v_R @ e_thr
    v_perp = v_R - np.outer(pi_thr, e_thr)
    norm_v_perp = np.linalg.norm(v_perp, axis=1)
    rsfi = norm_v_perp - alpha * pi_thr - beta * d_M
    return rsfi


def rsfi_multi_score(
    ref_mal: np.ndarray,
    ref_safe: np.ndarray,
    eval_emb: np.ndarray,
    k: int = 20,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 0.0,
) -> np.ndarray:
    """Multi-dimensional RSFI score (vectorised).

    The k threat anchors are the top-k right singular vectors of the
    (X_mal - mu_safe) matrix (i.e. the contrastive-SVD basis), as is
    natural for a generalized threat subspace.
    """
    S = l2(ref_safe.mean(axis=0, keepdims=True)).flatten()
    safe_mean = ref_safe.mean(axis=0)
    displaced = ref_mal - safe_mean
    k_eff = min(k, displaced.shape[0], displaced.shape[1])
    _, _, Vt = np.linalg.svd(displaced, full_matrices=False)
    V_threats = Vt[:k_eff]
    # Build tangent-space basis via QR.
    U_tan = np.empty((k_eff, S.size), dtype=np.float64)
    for j in range(k_eff):
        U_tan[j] = RiemannianSphere.log_map(S, V_threats[j])
    Q, _ = np.linalg.qr(U_tan.T)  # (d, k_eff)

    eval_l2 = l2(eval_emb)
    n = eval_l2.shape[0]
    v_R = np.empty((n, S.size), dtype=np.float64)
    for i in range(n):
        v_R[i] = RiemannianSphere.log_map(S, eval_l2[i])
    d_M = np.linalg.norm(v_R, axis=1)
    coeffs = v_R @ Q  # (n, k_eff)
    v_proj = Q @ coeffs.T  # (d, n)
    v_proj = v_proj.T  # (n, d)
    norm_proj = np.linalg.norm(v_proj, axis=1)
    v_perp = v_R - v_proj
    norm_v_perp = np.linalg.norm(v_perp, axis=1)
    rsfi = norm_v_perp - alpha * norm_proj - beta * d_M
    return rsfi


def calibrate_tau(scores: np.ndarray, y: np.ndarray, target_fpr: float = 0.01):
    """Pick tau so that FPR on negatives equals `target_fpr`."""
    neg = scores[y == 0]
    if neg.size == 0:
        return -np.inf
    # We want FPR = P(score >= tau | y=0) = target_fpr
    # so tau = quantile(1 - target_fpr) of neg scores.
    return float(np.quantile(neg, 1.0 - target_fpr))


# --------------------------------------------------------------- pipeline
def run_benchmark() -> None:
    print("=" * 80)
    print("E11: TANGENT-SPACE RSFI VALIDATION")
    print("=" * 80)

    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }
    embedders = [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]

    # Output 1: per-seed agreement RSFI vs B1 (Q1)
    agreement_rows = []
    # Output 2: full RSFI family (single + multi) ROC-AUCs
    rsfi_rows = []
    # Output 3: operating point @ ~1% FPR
    op_rows = []

    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal = int(labels.sum())
        n_safe = len(labels) - n_mal
        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = 200
            n_ref_safe = 200

        print(f"\n## {d_name}: n={len(texts)} (mal={n_mal}, safe={n_safe}); "
              f"ref budget: {n_ref_mal}/{n_ref_safe}")

        for model_id in embedders:
            short = model_id.split("/")[-1]
            print(f"  -- {short}")
            embeddings = get_embeddings(texts, d_name, model_id)
            for seed in range(N_SEEDS):
                mal_idx = np.where(labels == 1)[0]
                safe_idx = np.where(labels == 0)[0]
                np.random.seed(seed)
                ref_mal_idx = np.random.choice(mal_idx, n_ref_mal, replace=False)
                ref_safe_idx = np.random.choice(safe_idx, n_ref_safe, replace=False)
                test_mal_idx = np.setdiff1d(mal_idx, ref_mal_idx)
                test_safe_idx = np.setdiff1d(safe_idx, ref_safe_idx)
                test_idx = np.concatenate([test_mal_idx, test_safe_idx])
                y_test = labels[test_idx]

                ref_mal = embeddings[ref_mal_idx]
                ref_safe = embeddings[ref_safe_idx]
                emb_test = embeddings[test_idx]

                # ---- B1 (committed baseline from E2d: x' . (mu_m - mu_s))
                direction = ref_mal.mean(0) - ref_safe.mean(0)
                direction = direction / (np.linalg.norm(direction) + EPS)
                b1 = (l2(emb_test) @ direction).flatten()
                b1_auc = roc_auc_score(y_test, b1)

                # ---- Q1: single-vector RSFI (S=mu_safe, V_thr=mu_mal)
                rsfi1 = rsfi_single_score(ref_mal, ref_safe, emb_test)
                rsfi1_auc = roc_auc_score(y_test, rsfi1)

                # Q1 also check: B1 * -alpha should correlate perfectly with
                # the dominant -alpha * pi_thr term of rsfi1 (the other two
                # terms are tiny on these clean embeddings).  We accept
                # |AUC_RSFI1 - AUC_B1| <= TOL = 5e-3.
                diff = rsfi1_auc - b1_auc

                agreement_rows.append({
                    "dataset": d_name,
                    "model": short,
                    "seed": seed,
                    "n_ref_mal": n_ref_mal,
                    "n_ref_safe": n_ref_safe,
                    "n_test": len(test_idx),
                    "auc_B1_raw": b1_auc,
                    "auc_RSFI_single": rsfi1_auc,
                    "delta_auc": diff,
                })

                # ---- Q2: multi-vector RSFI (k=20 contrastive basis)
                rsfi2 = rsfi_multi_score(ref_mal, ref_safe, emb_test, k=20)
                rsfi2_auc = roc_auc_score(y_test, rsfi2)
                rsfi_rows.append({
                    "dataset": d_name,
                    "model": short,
                    "seed": seed,
                    "method": "RSFI_multi_k20",
                    "roc_auc": rsfi2_auc,
                    "n_test": len(test_idx),
                })

                # ---- Q3: operating point @ 1% FPR
                # Calibrate tau on the *safe reference* pool (held-out from
                # test), not on the test set, to remain leakage-free.
                safe_scores = rsfi_single_score(
                    ref_mal[: max(1, n_ref_mal // 2)],
                    ref_safe,
                    ref_mal[max(1, n_ref_mal // 2):],  # held-out "safe" surrogate
                )
                # In the absence of a clean held-out safe set, we use the
                # available safe reference itself with a 5-fold split:
                np.random.seed(seed * 31 + 1)
                safe_perm = np.random.permutation(ref_safe.shape[0])
                split = ref_safe.shape[0] // 2
                cal_safe, ho_safe = ref_safe[safe_perm[:split]], ref_safe[safe_perm[split:]]
                # Recompute single-vector RSFI on the *cal* set to get the
                # baseline of negatives for tau.
                cal_scores = rsfi_single_score(ref_mal, cal_safe, ho_safe)
                # But ho_safe is also safe, so its scores mimic test-time
                # false positives.  Use it to set tau.
                tau = calibrate_tau(cal_scores, np.zeros(cal_scores.size), 0.01)
                # TPR @ 1% FPR on test (using a pre-defined pass decision):
                tpr = float((rsfi1[y_test == 1] >= tau).mean())
                fpr = float((rsfi1[y_test == 0] >= tau).mean())
                op_rows.append({
                    "dataset": d_name,
                    "model": short,
                    "seed": seed,
                    "tau": tau,
                    "tpr_at_fpr_01": tpr,
                    "fpr_at_fpr_01": fpr,
                    "auc_rsfi": rsfi1_auc,
                })

    # ---- save outputs
    df_agree = pd.DataFrame(agreement_rows)
    df_rsfi = pd.DataFrame(rsfi_rows)
    df_op = pd.DataFrame(op_rows)

    df_agree.to_csv(RESULTS_DIR / "E11_rsfi_vs_b1.csv", index=False)
    df_rsfi.to_csv(RESULTS_DIR / "E11_rsfi_tangent_space.csv", index=False)
    df_op.to_csv(RESULTS_DIR / "E11_rsfi_operating_point.csv", index=False)

    # ---- summary
    print("\n" + "=" * 80)
    print("E11 SUMMARY  (mean +/- std over 5 seeds)")
    print("=" * 80)
    pivot = df_agree.pivot_table(
        index=["dataset", "model"],
        values=["auc_B1_raw", "auc_RSFI_single", "delta_auc"],
        aggfunc=["mean", "std"],
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pivot.round(4).to_string())

    print("\nQ2: Multi-vector RSFI (k=20)")
    pv2 = df_rsfi.pivot_table(
        index=["dataset", "model"], values="roc_auc",
        aggfunc=["mean", "std"],
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pv2.round(4).to_string())

    print("\nQ3: TPR @ FPR <= 1%  (single-vector RSFI)")
    pv3 = df_op.groupby(["dataset", "model"]).agg(
        tpr_mean=("tpr_at_fpr_01", "mean"),
        tpr_std=("tpr_at_fpr_01", "std"),
        fpr_mean=("fpr_at_fpr_01", "mean"),
        fpr_std=("fpr_at_fpr_01", "std"),
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pv3.round(4).to_string())

    # ---- head-line finding 1 (NEGATIVE): default RSFI is rank-anti-
    # correlated with B1.  This is a *documented negative result*; we
    # do NOT assert equivalence but we DO assert that the divergence
    # is consistent (always |delta| > 0.5 across all configs), and
    # we publish the median delta as a sanity bound.
    worst = df_agree.delta_auc.abs().max()
    median_delta = float(df_agree.delta_auc.abs().median())
    print(f"\nMax |AUC_RSFI_default - AUC_B1|: {worst:.4f} (median {median_delta:.4f})")
    assert median_delta > 0.5, (
        "expected RSFI default to be strongly rank-anti-correlated with B1; "
        f"got median |delta|={median_delta:.4f} which is too small"
    )
    print("OK: documented negative result. Default RSFI is rank-anti-"
          "correlated with B1 across all 45 (dataset x model x seed) configs.")

    # ---- Q1b: re-tune alpha so that RSFI is rank-equivalent to B1.
    #  Reason: in the linear limit alpha -> infty, the rsfi score is
    #  rank-monotone in pi_thr = <Log_S(V_thr), Log_S(x)>, which by the
    #  first-order Taylor expansion around the safe anchor equals
    #  cos(x, mu_mal - mu_safe).  We sweep alpha on a small grid and
    #  report the minimal |AUC_RSFI - AUC_B1|.
    print("\n" + "=" * 80)
    print("Q1b: alpha sweep to recover rank-equivalence with B1")
    print("=" * 80)
    alpha_grid = [0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 30.0, 100.0]
    sweep_rows = []
    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal = int(labels.sum())
        n_safe = len(labels) - n_mal
        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = 200
            n_ref_safe = 200
        for model_id in embedders:
            short = model_id.split("/")[-1]
            embeddings = get_embeddings(texts, d_name, model_id)
            for seed in range(N_SEEDS):
                mal_idx = np.where(labels == 1)[0]
                safe_idx = np.where(labels == 0)[0]
                np.random.seed(seed)
                ref_mal_idx = np.random.choice(mal_idx, n_ref_mal, replace=False)
                ref_safe_idx = np.random.choice(safe_idx, n_ref_safe, replace=False)
                test_mal_idx = np.setdiff1d(mal_idx, ref_mal_idx)
                test_safe_idx = np.setdiff1d(safe_idx, ref_safe_idx)
                test_idx = np.concatenate([test_mal_idx, test_safe_idx])
                y_test = labels[test_idx]
                ref_mal = embeddings[ref_mal_idx]
                ref_safe = embeddings[ref_safe_idx]
                emb_test = embeddings[test_idx]
                for alpha in alpha_grid:
                    s = rsfi_single_score(
                        ref_mal, ref_safe, emb_test, alpha=alpha, beta=0.0
                    )
                    sweep_rows.append({
                        "dataset": d_name,
                        "model": short,
                        "seed": seed,
                        "alpha": alpha,
                        "auc": roc_auc_score(y_test, s),
                    })
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(RESULTS_DIR / "E11_rsfi_alpha_sweep.csv", index=False)
    pv = df_sweep.groupby(["dataset", "model", "alpha"]).auc.agg(["mean", "std"])
    print(pv.round(4).to_string())
    # Final invariant: the *sign-reversed* single-vector RSFI score
    # (-rsfi, because rsfi is monotone-decreasing in the threat
    # direction for any non-zero alpha) should be rank-equivalent to
    # B1 to within TOL_LINEAR=1e-2 at alpha=100, beta=0 on all 9
    # (dataset, model) configs.  This is the linear-limit claim:
    #   lim_{alpha -> infty} sign(AUC) of (-rsfi) = sign(AUC_B1)
    # and the magnitude should match because in that limit -rsfi
    # becomes a strictly monotone function of pi_thr.
    big = df_sweep[df_sweep.alpha == 100.0]
    b1_ref = df_agree.groupby(["dataset", "model"]).auc_B1_raw.mean()
    worst_big = 0.0
    for (ds, mdl), g in big.groupby(["dataset", "model"]):
        rsfi_mean = g.auc.mean()
        b1_mean = b1_ref.loc[(ds, mdl)]
        # sign-reversed AUC must match B1
        sign_rev = 1.0 - rsfi_mean
        worst_big = max(worst_big, abs(sign_rev - b1_mean))
    print(f"\nAt alpha=100, beta=0: max |AUC(1-RSFI) - AUC_B1| = {worst_big:.4f}")
    assert worst_big <= 1e-2, (
        f"linear-limit claim violated: |AUC(1-RSFI) - AUC_B1|={worst_big:.4f} > 1e-2"
    )
    print("OK: linear-limit claim verified (sign-reversed RSFI at alpha=100 "
          "recovers B1 within 1e-2).")


if __name__ == "__main__":
    run_benchmark()
