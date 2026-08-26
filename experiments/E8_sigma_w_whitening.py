"""
E8_sigma_w_whitening.py
================================================================================
Reproducible implementation of the Sigma_T vs Sigma_W whitening comparison
(Report Table 4) and the 1-class / 2-class kNN comparison (Report Table 5).

Motivation (audit defect D1): the numbers previously existed only as console
text logs (rsfi_raw_audit_verification.txt). This script makes them reproducible
from the repository commit.

Methods evaluated per (dataset, embedder, seed):
  Group B (discriminant direction under different whiteners):
    - B1_raw          : discriminant mean direction in raw space
    - B1b_SigmaT_wh   : discriminant direction after ZCA whitening fitted on
                        the COMBINED reference pool (Sigma_T), as in E2d
    - B1w_SigmaW_wh   : discriminant direction after POOLED WITHIN-CLASS
                        whitening:
                          Sigma_W = 0.5 * (LW(C_mal) + LW(C_safe))
                        where C_mal / C_safe are Ledoit-Wolf-shrunk covariance
                        matrices of each reference class centered at its own
                        class mean, and centering for the transform uses the
                        combined reference mean mu = 0.5*(mu_mal + mu_safe).
  Group K (kNN baselines, cosine similarity, k=5):
    - kNN_1class      : mean top-5 cosine similarity to MALICIOUS references
                        only (memory: N_ref_mal vectors)
    - kNN_2class      : mean top-5 sim to malicious MINUS mean top-5 sim to
                        safe references (contrastive; memory: N_ref total)

Outputs:
  data/results/E8_sigma_w.csv   (per-seed AUC/PR-AUC for group B)
  data/results/E8_knn.csv       (per-seed AUC/PR-AUC for group K)
  data/results/E8_delong_tests.csv (per-seed DeLong p-values for key pairs)

Notes:
  - Dataset loaders and embedding cache are reused verbatim from
    E2d_safe_aware_multidataset.py to guarantee identical splits/data.
  - Known inconsistency inherited from E2d (documented, not copied here):
    E2d's score_contrastive_svd does NOT L2-normalize emb_eval before
    projection. This script normalizes everywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_wild,
    load_xstest,
    get_embeddings,
    delong_test,
)
from rsfi.whitening import SphericalWhitening  # noqa: E402

N_SEEDS = 5
KNN_K = 5


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-15)


# --- Whitening variants -------------------------------------------------------

def fit_sigma_t_whitener(ref_comb: np.ndarray, dim: int) -> SphericalWhitening:
    """Total-covariance spherical ZCA whitening (identical to E2d protocol)."""
    wh = SphericalWhitening(dim=dim)
    wh.fit(ref_comb)
    return wh


class PooledWithinClassWhitening:
    """
    Whitening by pooled within-class covariance:
        Sigma_W = 0.5 * (LW(C_mal) + LW(C_safe))
    Each class covariance is estimated with Ledoit-Wolf shrinkage on samples
    centered at its own class mean. Transform centers by the combined
    reference mean and re-projects onto the unit sphere (same as
    SphericalWhitening.transform).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.mu = None
        self.W = None

    def fit(self, ref_mal: np.ndarray, ref_safe: np.ndarray):
        mu_mal = ref_mal.mean(axis=0)
        mu_safe = ref_safe.mean(axis=0)
        self.mu = 0.5 * (mu_mal + mu_safe)

        lw_mal = LedoitWolf().fit(ref_mal - mu_mal)
        lw_safe = LedoitWolf().fit(ref_safe - mu_safe)
        sigma_w = 0.5 * (lw_mal.covariance_ + lw_safe.covariance_)

        evals, evecs = np.linalg.eigh(sigma_w)
        evals = np.maximum(evals, 1e-12)
        self.W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return l2norm((x - self.mu) @ self.W)


# --- Scorers ------------------------------------------------------------------

def score_discriminant(emb_eval: np.ndarray,
                       ref_mal: np.ndarray,
                       ref_safe: np.ndarray) -> np.ndarray:
    direction = ref_mal.mean(axis=0) - ref_safe.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-15)
    return (l2norm(emb_eval) @ direction).flatten()


def knn_scores(emb_eval_n: np.ndarray, ref_mal_n: np.ndarray,
               ref_safe_n: np.ndarray, k: int = KNN_K):
    """Cosine-similarity kNN scores. Inputs must be L2-normalized."""
    k_mal = min(k, ref_mal_n.shape[0])
    k_safe = min(k, ref_safe_n.shape[0])

    sim_mal = emb_eval_n @ ref_mal_n.T
    top_mal = np.sort(sim_mal, axis=1)[:, -k_mal:].mean(axis=1)

    sim_safe = emb_eval_n @ ref_safe_n.T
    top_safe = np.sort(sim_safe, axis=1)[:, -k_safe:].mean(axis=1)

    knn_1class = top_mal
    knn_2class = top_mal - top_safe
    return knn_1class, knn_2class


# --- Benchmark loop -----------------------------------------------------------

def run():
    print("=" * 80)
    print("E8: Sigma_T vs Sigma_W whitening + 1-class/2-class kNN")
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

    rows_b, rows_k, rows_d = [], [], []

    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal, n_safe = int(labels.sum()), len(labels) - int(labels.sum())
        print(f"\n### {d_name}: {len(texts)} items ({n_mal} mal / {n_safe} safe)")

        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = n_ref_safe = 200
        print(f"    Budget: n_ref_mal={n_ref_mal}, n_ref_safe={n_ref_safe}")

        for model_id in embedders:
            model_short = model_id.split("/")[-1]
            embeddings = get_embeddings(texts, d_name, model_id)
            dim = embeddings.shape[1]
            print(f"  --- {model_short} (d={dim}) ---")

            for seed in range(N_SEEDS):
                mal_idx = np.where(labels == 1)[0]
                safe_idx = np.where(labels == 0)[0]

                np.random.seed(seed)
                ref_mal_idx = np.random.choice(mal_idx, size=n_ref_mal, replace=False)
                ref_safe_idx = np.random.choice(safe_idx, size=n_ref_safe, replace=False)

                test_idx = np.concatenate([
                    np.setdiff1d(mal_idx, ref_mal_idx),
                    np.setdiff1d(safe_idx, ref_safe_idx),
                ])
                y_test = labels[test_idx]

                emb_test = embeddings[test_idx]
                ref_mal_raw = embeddings[ref_mal_idx]
                ref_safe_raw = embeddings[ref_safe_idx]
                ref_comb = np.vstack([ref_mal_raw, ref_safe_raw])

                # Whiteners
                wh_t = fit_sigma_t_whitener(ref_comb, dim)
                wh_w = PooledWithinClassWhitening(dim).fit(ref_mal_raw, ref_safe_raw)

                # Transformed spaces
                test_t = wh_t.transform(emb_test)
                mal_t = wh_t.transform(ref_mal_raw)
                safe_t = wh_t.transform(ref_safe_raw)

                test_w = wh_w.transform(emb_test)
                mal_w = wh_w.transform(ref_mal_raw)
                safe_w = wh_w.transform(ref_safe_raw)

                # Group B: discriminant direction under each whitener
                scores_b = {
                    "B1_raw": score_discriminant(emb_test, ref_mal_raw, ref_safe_raw),
                    "B1b_SigmaT_wh": score_discriminant(test_t, mal_t, safe_t),
                    "B1w_SigmaW_wh": score_discriminant(test_w, mal_w, safe_w),
                }

                # Group K: kNN baselines (in raw normalized space, as in E7)
                knn1, knn2 = knn_scores(
                    l2norm(emb_test), l2norm(ref_mal_raw), l2norm(ref_safe_raw)
                )
                scores_k = {"kNN_1class": knn1, "kNN_2class": knn2}

                base = {
                    "dataset": d_name, "model": model_short, "dim": dim,
                    "seed": seed, "n_ref_mal": n_ref_mal, "n_ref_safe": n_ref_safe,
                    "n_test": len(test_idx),
                }
                for m, sc in {**scores_b, **scores_k}.items():
                    row = dict(base, method=m,
                               roc_auc=roc_auc_score(y_test, sc),
                               pr_auc=average_precision_score(y_test, sc))
                    (rows_b if m in scores_b else rows_k).append(row)

                # DeLong pairs of interest
                for m1, m2, tag in [
                    ("B1w_SigmaW_wh", "B1b_SigmaT_wh", "B1w_vs_B1b"),
                    ("B1b_SigmaT_wh", "B1_raw", "B1b_vs_B1"),
                    ("B1w_SigmaW_wh", "B1_raw", "B1w_vs_B1"),
                ]:
                    rows_d.append(dict(
                        base, pair=tag, method_1=m1, method_2=m2,
                        auc_diff=float(roc_auc_score(y_test, scores_b[m1])
                                       - roc_auc_score(y_test, scores_b[m2])),
                        p_value=delong_test(y_test, scores_b[m1], scores_b[m2]),
                    ))

    out_dir = Path(__file__).parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_b = pd.DataFrame(rows_b)
    df_k = pd.DataFrame(rows_k)
    df_d = pd.DataFrame(rows_d)
    df_b.to_csv(out_dir / "E8_sigma_w.csv", index=False)
    df_k.to_csv(out_dir / "E8_knn.csv", index=False)
    df_d.to_csv(out_dir / "E8_delong_tests.csv", index=False)

    print("\n=== GROUP B summary (ROC-AUC mean +- std over seeds) ===")
    print(df_b.pivot_table(index=["dataset", "model"], columns="method",
                           values="roc_auc").round(4).to_string())
    print("\n=== GROUP K summary ===")
    print(df_k.pivot_table(index=["dataset", "model"], columns="method",
                           values="roc_auc").round(4).to_string())
    print("\nSaved:", out_dir / "E8_sigma_w.csv")
    print("Saved:", out_dir / "E8_knn.csv")
    print("Saved:", out_dir / "E8_delong_tests.csv")


if __name__ == "__main__":
    run()