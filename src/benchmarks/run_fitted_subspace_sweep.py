"""
Fitted RSFI Subspace Dimension and Reference Size Sweep.
Sweeps k in [1, 5, 10, 20, 30, 40] and N_ref in [50, 100, 200] to find peak ROC-AUC.
"""

import os
from pathlib import Path

CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsfi import RiemannianSphere, SphericalWhitening, MultiDimensionalRSFIFilter


def run_subspace_sweep(
    model_name: str = "all-mpnet-base-v2", n_samples_per_class: int = 1500
):
    print("=" * 90)
    print("   RSFI DATA-DRIVEN SUBSPACE DIMENSION (k) AND REFERENCE SIZE SWEEP")
    print("=" * 90 + "\n")

    model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
    dim = model.get_sentence_embedding_dimension()

    from rsfi.benchmarks.wildchat_10k import WildChatBenchmarkRunner

    runner = WildChatBenchmarkRunner(model_name=model_name, cache_folder=CACHE_DIR)
    samples = runner.load_dataset_samples(target_per_class=n_samples_per_class)

    np.random.seed(42)
    perm = np.random.permutation(len(samples))
    samples = [samples[i] for i in perm]
    y_true = np.array([1 if s.scenario_type == "MALICIOUS" else 0 for s in samples])

    all_texts = [s.text for s in samples]
    print(f"[ENCODING] Encoding {len(all_texts)} dataset prompts with {model_name}...")
    raw_embeddings = model.encode(
        all_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True
    )

    mal_indices = np.where(y_true == 1)[0]
    safe_indices = np.where(y_true == 0)[0]

    sweep_results = []

    for n_ref in [50, 100, 200]:
        ref_mal_idx = mal_indices[:n_ref]
        ref_safe_idx = safe_indices[:n_ref]
        test_idx = np.concatenate([mal_indices[n_ref:], safe_indices[n_ref:]])

        test_raw = raw_embeddings[test_idx]
        y_test = y_true[test_idx]

        # Fit ZCA Whitening ONLY on reference safe set (no test leakage)
        calib_raw = raw_embeddings[ref_safe_idx]
        whitening = SphericalWhitening(dim=dim)
        whitening.fit(RiemannianSphere.normalize(calib_raw))

        # System Anchor S
        ref_safe_whitened = whitening.transform(raw_embeddings[ref_safe_idx])
        S = RiemannianSphere.normalize(
            np.mean(ref_safe_whitened, axis=0, keepdims=True)
        )[0]

        # Tangent Threat Vectors
        ref_mal_whitened = whitening.transform(raw_embeddings[ref_mal_idx])
        tangent_threats = np.array(
            [
                RiemannianSphere.log_map(S, ref_mal_whitened[i])
                for i in range(len(ref_mal_whitened))
            ]
        )

        # Perform SVD
        U, Sigma, Vh = np.linalg.svd(tangent_threats, full_matrices=False)

        test_whitened = whitening.transform(test_raw)

        # Split test into val and final test for honest k selection
        n_test = len(test_idx)
        val_size = min(200, n_test // 3)
        val_idx_local = np.arange(val_size)
        final_test_idx_local = np.arange(val_size, n_test)

        y_val = y_test[val_idx_local]
        y_final_test = y_test[final_test_idx_local]

        # Find best k on validation set
        best_k = None
        best_val_auc = 0.0
        k_val_aucs = {}

        for k in [1, 5, 10, 20, 30, 40]:
            fitted_threat_vectors = Vh[:k]
            rsfi_filter = MultiDimensionalRSFIFilter(
                S, fitted_threat_vectors, alpha=1.5, beta=0.2, tau=0.5, is_tangent=True
            )

            val_scores = []
            for i in val_idx_local:
                res = rsfi_filter.evaluate(test_whitened[i])
                val_scores.append(-res["rsfi"])

            val_auc = float(roc_auc_score(y_val, np.array(val_scores)))
            k_val_aucs[k] = val_auc

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_k = k

        # Evaluate best k on final test set
        fitted_threat_vectors = Vh[:best_k]
        rsfi_filter = MultiDimensionalRSFIFilter(
            S, fitted_threat_vectors, alpha=1.5, beta=0.2, tau=0.5, is_tangent=True
        )

        rsfi_scores = []
        t0 = time.perf_counter()
        for i in final_test_idx_local:
            res = rsfi_filter.evaluate(test_whitened[i])
            rsfi_scores.append(-res["rsfi"])
        lat_ms = (time.perf_counter() - t0) * 1000.0 / len(final_test_idx_local)

        auc_score = float(roc_auc_score(y_final_test, np.array(rsfi_scores)))
        k = best_k
        var_exp = float(np.sum(Sigma[:k] ** 2) / np.sum(Sigma**2) * 100.0)

            sweep_results.append(
                {
                    "N_ref": n_ref,
                    "k_dim": k,
                    "roc_auc": auc_score,
                    "explained_var_pct": var_exp,
                    "latency_us": lat_ms * 1000.0,
                    "best_k_on_val": best_k,
                    "val_auc": best_val_auc,
                }
            )
            print(
                f"N_ref={n_ref:3d} | best_k={best_k:2d} (val_auc={best_val_auc:.4f}) | "
                f"Test ROC-AUC = {auc_score:.4f} | Latency = {lat_ms*1000.0:.1f} us"
            )

    df_res = pd.DataFrame(sweep_results)
    out_dir = Path("data/reports")
    df_res.to_csv(out_dir / "fitted_subspace_sweep.csv", index=False)
    print(
        f"\n[EXPORT] Sweep results saved to: {out_dir / 'fitted_subspace_sweep.csv'}\n"
    )


if __name__ == "__main__":
    run_subspace_sweep(model_name="all-mpnet-base-v2", n_samples_per_class=1500)
