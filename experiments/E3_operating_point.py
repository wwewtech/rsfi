"""
E3: Operating Point Experiment
================================
For guardrails, the critical metric is TPR at constrained FPR (e.g., 1% or 0.1%).
This is more realistic than overall AUC: in production, we calibrate a threshold
on validation to achieve target FPR, then measure TPR on test.

Metrics computed:
- ROC-AUC (for comparison with literature)
- PR-AUC (precision-recall, standard in GradSafe/safety research)
- TPR @ FPR ≤ 1% (catch rate with 1% false positive budget)
- TPR @ FPR ≤ 0.1% (catch rate with 0.1% false positive budget)

Protocol:
1. Calibrate threshold on validation set to achieve target FPR
2. Apply that threshold to test set and measure TPR
3. Report confidence intervals via bootstrap
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.stats import bootstrap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening


def find_threshold_for_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float
) -> float:
    """
    Find the threshold that achieves target FPR on validation set.

    Args:
        y_true: Ground truth labels (1 = positive/malicious, 0 = negative/safe)
        scores: Scores (higher = more likely positive)
        target_fpr: Target false positive rate (e.g., 0.01 for 1%)

    Returns:
        threshold: Score threshold to achieve target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Find the threshold where FPR is closest to (but not exceeding) target
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        # No threshold achieves target FPR; use most conservative
        return thresholds.max()

    # Among valid thresholds, pick the one with highest TPR (lowest threshold)
    best_idx = valid_idx[-1]
    return thresholds[best_idx]


def compute_tpr_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """
    Compute TPR (true positive rate) at a given threshold.

    TPR = TP / (TP + FN) = fraction of positives correctly identified
    """
    predictions = scores >= threshold

    positives = y_true == 1
    true_positives = (predictions & positives).sum()

    tpr = true_positives / positives.sum() if positives.sum() > 0 else 0.0
    return float(tpr)


def compute_rsfi_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    k: int = 20,
    apply_whitening: bool = True
) -> np.ndarray:
    """Compute RSFI scores using SVD-based subspace projection."""
    ref_emb = embeddings[ref_indices]

    # Apply whitening
    if apply_whitening:
        wh = SphericalWhitening(dim=embeddings.shape[1])
        wh.fit(ref_emb)
        embeddings_wh = wh.transform(embeddings)
        ref_emb_wh = embeddings_wh[ref_indices]
    else:
        embeddings_wh = embeddings
        ref_emb_wh = ref_emb

    # Normalize
    ref_emb_wh = ref_emb_wh / (np.linalg.norm(ref_emb_wh, axis=1, keepdims=True) + 1e-15)
    embeddings_wh = embeddings_wh / (np.linalg.norm(embeddings_wh, axis=1, keepdims=True) + 1e-15)

    # SVD on reference set
    U, S, Vt = np.linalg.svd(ref_emb_wh.T, full_matrices=False)
    U_k = U[:, :k]

    # Project onto top-k subspace
    proj = embeddings_wh @ U_k
    scores = np.linalg.norm(proj, axis=1)

    return scores


def compute_naive_cosine_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray
) -> np.ndarray:
    """Compute naive cosine similarity to mean of reference set."""
    ref_emb = embeddings[ref_indices]
    ref_mean = np.mean(ref_emb, axis=0, keepdims=True)
    ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-15)

    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-15)
    scores = embeddings_norm @ ref_mean.T
    return scores.flatten()


def load_dataset() -> Tuple[List[str], List[int]]:
    """Load dataset for evaluation (try wild dataset first)."""
    print("Loading dataset...")

    # Try wild dataset
    results_path = Path(__file__).parent.parent / "data" / "data/results" / "sfi_wild_10k_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        texts = df['text'].tolist()
        labels = (df['scenario_type'] == 'MALICIOUS').astype(int).tolist()
        print(f"  Loaded wild dataset: {len(texts)} examples ({sum(labels)} malicious)")
        return texts, labels

    # Try to load from datasets library
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

        print(f"  Loaded ToxicChat: {len(texts)} examples ({sum(labels)} malicious)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return [], []


def evaluate_method(
    method_name: str,
    score_fn: Callable,
    embeddings: np.ndarray,
    labels: np.ndarray,
    ref_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    target_fprs: List[float] = [0.01, 0.001]
) -> Dict:
    """
    Evaluate a single method with operating point metrics.

    Returns:
        Dictionary with ROC-AUC, PR-AUC, TPR@FPR for each target FPR
    """
    # Compute scores
    all_idx = np.concatenate([ref_idx, val_idx, test_idx])
    all_embeddings = embeddings[all_idx]

    scores_all = score_fn(all_embeddings, np.arange(len(ref_idx)))

    # Split scores
    n_ref = len(ref_idx)
    n_val = len(val_idx)

    scores_val = scores_all[n_ref:n_ref+n_val]
    scores_test = scores_all[n_ref+n_val:]

    y_val = labels[val_idx]
    y_test = labels[test_idx]

    # Compute standard metrics on test set
    roc_auc = roc_auc_score(y_test, scores_test)
    pr_auc = average_precision_score(y_test, scores_test)

    result = {
        'method': method_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    # Compute TPR at each target FPR
    for target_fpr in target_fprs:
        # Calibrate threshold on validation set
        threshold = find_threshold_for_fpr(y_val, scores_val, target_fpr)

        # Measure actual FPR on validation (sanity check)
        val_predictions = scores_val >= threshold
        val_fpr = ((val_predictions) & (y_val == 0)).sum() / (y_val == 0).sum()

        # Compute TPR on test set with this threshold
        tpr_test = compute_tpr_at_threshold(y_test, scores_test, threshold)

        # Compute actual FPR on test set (for reporting)
        test_predictions = scores_test >= threshold
        test_fpr = ((test_predictions) & (y_test == 0)).sum() / (y_test == 0).sum()

        fpr_key = f"fpr_{int(target_fpr*10000):04d}"  # e.g., fpr_0100 for 1%
        result[f'tpr_at_{fpr_key}'] = tpr_test
        result[f'actual_{fpr_key}_test'] = test_fpr
        result[f'threshold_for_{fpr_key}'] = threshold

    return result


def main():
    """Run E3 operating point experiment."""
    print("="*80)
    print("E3: OPERATING POINT EXPERIMENT")
    print("="*80)

    # Configuration
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    N_REF = 200
    N_VAL = 200
    N_SEEDS = 5
    TARGET_FPRS = [0.01, 0.001]  # 1% and 0.1%

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  N_ref: {N_REF}")
    print(f"  N_val: {N_VAL}")
    print(f"  N_seeds: {N_SEEDS}")
    print(f"  Target FPRs: {TARGET_FPRS}")

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Load dataset
    texts, labels = load_dataset()
    if len(texts) == 0:
        print("\nERROR: No dataset loaded. Cannot run experiment.")
        return

    # Encode all texts
    print("\nEncoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    labels = np.array(labels)

    # Run evaluation across seeds
    results = []

    for seed in range(N_SEEDS):
        print(f"\nSeed {seed+1}/{N_SEEDS}")

        # Split data
        malicious_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]

        if len(malicious_idx) < N_REF + 100:
            print(f"  WARNING: Not enough malicious examples ({len(malicious_idx)})")
            continue

        if len(safe_idx) < N_VAL + 100:
            print(f"  WARNING: Not enough safe examples ({len(safe_idx)})")
            continue

        # Reference: N_REF malicious
        np.random.seed(seed)
        ref_idx = np.random.choice(malicious_idx, size=N_REF, replace=False)
        remaining_mal_idx = np.setdiff1d(malicious_idx, ref_idx)

        # Validation: N_VAL balanced
        val_mal_idx = np.random.choice(remaining_mal_idx, size=N_VAL//2, replace=False)
        val_safe_idx = np.random.choice(safe_idx, size=N_VAL//2, replace=False)
        val_idx = np.concatenate([val_mal_idx, val_safe_idx])

        # Test: everything else
        test_idx = np.setdiff1d(np.arange(len(texts)), np.concatenate([ref_idx, val_idx]))

        print(f"  Splits: ref={len(ref_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Method 1: RSFI-SVD
        result_rsfi = evaluate_method(
            "RSFI-SVD",
            lambda emb, ref: compute_rsfi_scores(emb, ref, k=20, apply_whitening=True),
            embeddings, labels, ref_idx, val_idx, test_idx, TARGET_FPRS
        )
        result_rsfi['seed'] = seed
        results.append(result_rsfi)

        # Method 2: Naive cosine
        result_cosine = evaluate_method(
            "naive_cosine",
            compute_naive_cosine_scores,
            embeddings, labels, ref_idx, val_idx, test_idx, TARGET_FPRS
        )
        result_cosine['seed'] = seed
        results.append(result_cosine)

        # Method 3: LogReg
        def logreg_score_fn(emb, ref_indices):
            # Train on ref + val
            train_idx_local = np.concatenate([
                np.arange(len(ref_indices)),
                len(ref_indices) + np.arange(len(val_idx))
            ])

            X_train = emb[train_idx_local]
            y_train = np.concatenate([
                np.ones(len(ref_indices)),
                labels[val_idx]
            ])

            logreg = LogisticRegression(max_iter=1000, random_state=seed)
            logreg.fit(X_train, y_train)

            return logreg.decision_function(emb)

        result_logreg = evaluate_method(
            "LogReg",
            logreg_score_fn,
            embeddings, labels, ref_idx, val_idx, test_idx, TARGET_FPRS
        )
        result_logreg['seed'] = seed
        results.append(result_logreg)

        print(f"  RSFI ROC-AUC: {result_rsfi['roc_auc']:.4f}, TPR@1%: {result_rsfi.get('tpr_at_fpr_0100', 0):.4f}")
        print(f"  Cosine ROC-AUC: {result_cosine['roc_auc']:.4f}, TPR@1%: {result_cosine.get('tpr_at_fpr_0100', 0):.4f}")
        print(f"  LogReg ROC-AUC: {result_logreg['roc_auc']:.4f}, TPR@1%: {result_logreg.get('tpr_at_fpr_0100', 0):.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data/results" / "E3_operating_point_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    summary_cols = ['roc_auc', 'pr_auc', 'tpr_at_fpr_0100', 'tpr_at_fpr_0010']
    summary = results_df.groupby('method')[summary_cols].agg(['mean', 'std'])

    print("\nMean ± Std across seeds:")
    print(summary.to_string())

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("TPR@FPR≤1% is the key metric for production guardrails.")
    print("This shows how many attacks we catch with a 1% false positive budget.")
    print("Compare methods at fixed FPR, not just overall AUC.")


if __name__ == "__main__":
    main()
