"""
E5: Whitening Stability in High Dimensions
===========================================
Hypothesis: At d=4096 and N_calib=200, the covariance matrix is rank-deficient
(rank ≤ min(N_calib-1, d) = 199). ZCA whitening should become more effective
when N_calib > d.

This experiment:
1. Sweep N_calib from 200 to 10,000
2. Measure condition number of covariance matrix
3. Measure AUC improvement from ZCA whitening
4. Test on high-dimensional embeddings (4096d from Qwen3-Embedding-8B)

Expected outcome: ZCA contribution should grow with N_calib when d is large.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening


def compute_rsfi_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    k: int = 20,
    apply_whitening: bool = True
) -> np.ndarray:
    """Compute RSFI scores with optional whitening."""
    ref_emb = embeddings[ref_indices]

    if apply_whitening:
        wh = SphericalWhitening(dim=embeddings.shape[1])
        wh.fit(ref_emb)
        embeddings_wh = wh.transform(embeddings)
        ref_emb_wh = embeddings_wh[ref_indices]

        # Compute condition number
        cov = np.cov(ref_emb.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero
        condition_number = eigenvalues.max() / eigenvalues.min() if len(eigenvalues) > 0 else np.inf
    else:
        embeddings_wh = embeddings
        ref_emb_wh = ref_emb
        condition_number = None

    # Normalize
    ref_emb_wh = ref_emb_wh / (np.linalg.norm(ref_emb_wh, axis=1, keepdims=True) + 1e-15)
    embeddings_wh = embeddings_wh / (np.linalg.norm(embeddings_wh, axis=1, keepdims=True) + 1e-15)

    # SVD
    U, S, Vt = np.linalg.svd(ref_emb_wh.T, full_matrices=False)
    U_k = U[:, :k]

    # Project
    proj = embeddings_wh @ U_k
    scores = np.linalg.norm(proj, axis=1)

    return scores, condition_number


def load_dataset() -> Tuple[List[str], List[int]]:
    """Load dataset for evaluation."""
    print("Loading dataset...")

    # Try wild dataset
    results_path = Path(__file__).parent.parent / "data" / "results" / "sfi_wild_10k_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        texts = df['text'].tolist()
        labels = df['is_blocked'].astype(int).tolist()
        print(f"  Loaded wild dataset: {len(texts)} examples")
        return texts, labels

    # Try ToxicChat
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

        texts = []
        labels = []

        for item in ds['train']:
            text = item.get('user_input', '')
            if not text:
                continue

            toxicity = item.get('toxicity', 0)
            jailbreaking = item.get('jailbreaking', 0)

            if toxicity == 1 or jailbreaking == 1:
                labels.append(1)
            elif toxicity == 0 and jailbreaking == 0:
                labels.append(0)
            else:
                continue

            texts.append(text)

        print(f"  Loaded ToxicChat: {len(texts)} examples")
        return texts, labels

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return [], []


def main():
    """Run E5 whitening stability experiment."""
    print("="*80)
    print("E5: WHITENING STABILITY IN HIGH DIMENSIONS")
    print("="*80)

    # Configuration
    N_CALIB_VALUES = [200, 500, 1000, 2000, 5000, 10000]
    N_SEEDS = 3

    # Try multiple models (prefer high-dimensional)
    MODELS = [
        ("sentence-transformers/all-mpnet-base-v2", 768),
        ("BAAI/bge-base-en-v1.5", 768),
    ]

    # Note: Qwen3-Embedding-8B requires special handling
    # Add it if available
    try:
        print("\nChecking for high-dimensional models...")
        test_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        dim = test_model.get_sentence_embedding_dimension()
        MODELS.append(("Alibaba-NLP/gte-Qwen2-1.5B-instruct", dim))
        print(f"  Found high-dim model: {dim}d")
    except Exception as e:
        print(f"  High-dim model not available: {e}")

    print(f"\nConfiguration:")
    print(f"  N_calib values: {N_CALIB_VALUES}")
    print(f"  N_seeds: {N_SEEDS}")
    print(f"  Models: {[m[0] for m in MODELS]}")

    # Load dataset
    texts, labels = load_dataset()
    if len(texts) == 0:
        print("\nERROR: No dataset loaded.")
        return

    labels = np.array(labels)

    # Run experiments for each model
    all_results = []

    for model_name, expected_dim in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name} ({expected_dim}d)")
        print('='*80)

        try:
            model = SentenceTransformer(model_name)
            actual_dim = model.get_sentence_embedding_dimension()

            if actual_dim != expected_dim:
                print(f"  WARNING: Expected {expected_dim}d, got {actual_dim}d")

            print(f"  Encoding {len(texts)} texts...")
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        # Get safe corpus for whitening calibration
        safe_idx = np.where(labels == 0)[0]
        malicious_idx = np.where(labels == 1)[0]

        if len(safe_idx) < max(N_CALIB_VALUES):
            print(f"  WARNING: Not enough safe examples ({len(safe_idx)})")
            # We don't continue here, we let the inner loop skip only the n_calib values that are too large.

        for n_calib in N_CALIB_VALUES:
            print(f"\n  N_calib = {n_calib}")

            if len(safe_idx) < n_calib:
                print(f"    Skipping (not enough safe examples)")
                continue

            for seed in range(N_SEEDS):
                np.random.seed(seed)

                # Sample calibration corpus from safe examples
                calib_idx = np.random.choice(safe_idx, size=n_calib, replace=False)

                # Reference set: 200 malicious
                n_ref = 200
                if len(malicious_idx) < n_ref + 100:
                    print(f"    WARNING: Not enough malicious examples")
                    break

                ref_idx = np.random.choice(malicious_idx, size=n_ref, replace=False)
                remaining_mal = np.setdiff1d(malicious_idx, ref_idx)

                # Test: remaining examples
                test_idx = np.setdiff1d(
                    np.arange(len(texts)),
                    np.concatenate([calib_idx, ref_idx])
                )

                y_test = labels[test_idx]

                # Combine ref + test for scoring
                eval_idx = np.concatenate([ref_idx, test_idx])

                # Score WITHOUT whitening
                scores_no_wh, _ = compute_rsfi_scores(
                    embeddings[eval_idx],
                    np.arange(n_ref),
                    k=20,
                    apply_whitening=False
                )
                scores_no_wh_test = scores_no_wh[n_ref:]
                auc_no_wh = roc_auc_score(y_test, scores_no_wh_test)

                # Score WITH whitening (calibrated on calib_idx)
                # Need to fit whitening on calib set, then transform eval set
                wh = SphericalWhitening(dim=actual_dim)
                wh.fit(embeddings[calib_idx])

                embeddings_eval_wh = wh.transform(embeddings[eval_idx])
                ref_emb_wh = embeddings_eval_wh[:n_ref]

                # Compute condition number of calibration covariance
                cov = np.cov(embeddings[calib_idx].T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                condition_number = eigenvalues.max() / eigenvalues.min() if len(eigenvalues) > 0 else np.inf
                rank_estimate = np.sum(eigenvalues > 1e-6)

                # Normalize and SVD
                ref_emb_wh = ref_emb_wh / (np.linalg.norm(ref_emb_wh, axis=1, keepdims=True) + 1e-15)
                embeddings_eval_wh = embeddings_eval_wh / (np.linalg.norm(embeddings_eval_wh, axis=1, keepdims=True) + 1e-15)

                U, S, Vt = np.linalg.svd(ref_emb_wh.T, full_matrices=False)
                U_k = U[:, :20]

                proj = embeddings_eval_wh @ U_k
                scores_wh = np.linalg.norm(proj, axis=1)
                scores_wh_test = scores_wh[n_ref:]
                auc_wh = roc_auc_score(y_test, scores_wh_test)

                # Record results
                all_results.append({
                    'model': model_name,
                    'dim': actual_dim,
                    'n_calib': n_calib,
                    'seed': seed,
                    'condition_number': condition_number,
                    'rank_estimate': rank_estimate,
                    'auc_no_whitening': auc_no_wh,
                    'auc_with_whitening': auc_wh,
                    'whitening_gain': auc_wh - auc_no_wh,
                    'n_test': len(test_idx)
                })

                print(f"    Seed {seed}: cond={condition_number:.1e}, rank≈{rank_estimate}, "
                      f"AUC: {auc_no_wh:.4f}→{auc_wh:.4f} (Δ={auc_wh-auc_no_wh:+.4f})")

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path(__file__).parent.parent / "results" / "E5_whitening_stability.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Whitening Gain vs N_calib")
    print("="*80)

    if results_df.empty:
        print("No results were generated.")
        return

    summary = results_df.groupby(['model', 'dim', 'n_calib'])[
        ['condition_number', 'whitening_gain']
    ].agg(['mean', 'std'])

    print(summary.to_string())

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("Expected: When N_calib < d, covariance is rank-deficient (high condition number).")
    print("ZCA should help more when N_calib ≥ d (full-rank covariance).")
    print("If whitening_gain is negative or zero for all N_calib, ZCA may hurt on this data.")


if __name__ == "__main__":
    main()
