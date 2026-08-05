"""
E2: Homogeneous Datasets Experiment
====================================
Critical test: evaluate RSFI on stylistically homogeneous datasets where
simple heuristics (text length, keyword matching) won't work.

Datasets:
1. ToxicChat-0124 (LMSYS) - diverse toxic/jailbreak vs clean conversations
2. XSTest-v2 - contrastive pairs ("kill the process" vs "kill the person")
3. Current wild dataset (for comparison)

Expected outcome: All methods will drop in performance, but if RSFI drops
LESS than naive cosine, that demonstrates the SVD contribution.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening


def load_toxicchat() -> Tuple[List[str], List[int]]:
    """Load ToxicChat-0124 dataset from LMSYS."""
    print("Loading ToxicChat-0124...")
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

        texts = []
        labels = []

        # Process train split
        for item in ds['train']:
            text = item.get('user_input', '')
            if not text:
                continue

            toxicity = item.get('toxicity', 0)
            jailbreaking = item.get('jailbreaking', 0)

            # Label as malicious if toxic OR jailbreaking
            if toxicity == 1 or jailbreaking == 1:
                labels.append(1)
            elif toxicity == 0 and jailbreaking == 0:
                labels.append(0)
            else:
                continue  # Skip unclear cases

            texts.append(text)

        print(f"  Loaded {len(texts)} examples ({sum(labels)} malicious, {len(labels)-sum(labels)} safe)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading ToxicChat: {e}")
        print("  Skipping ToxicChat dataset")
        return [], []


def load_xstest() -> Tuple[List[str], List[int]]:
    """
    Load XSTest-v2 contrastive pairs.
    This is the KILLER test for geometric methods: high cosine similarity
    but opposite safety labels.
    """
    print("Loading XSTest-v2...")
    try:
        # Try to load from HuggingFace
        ds = load_dataset("paul-rottger/exaggerated-safety-v01", split="test")

        texts = []
        labels = []

        for item in ds:
            prompt = item.get('prompt', '')
            label_str = item.get('label', '')

            if not prompt:
                continue

            # label: "safe" or "unsafe"
            if label_str == "unsafe":
                labels.append(1)
            elif label_str == "safe":
                labels.append(0)
            else:
                continue

            texts.append(prompt)

        print(f"  Loaded {len(texts)} examples ({sum(labels)} unsafe, {len(labels)-sum(labels)} safe)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading XSTest: {e}")
        print("  Skipping XSTest dataset")
        return [], []


def load_wild_dataset() -> Tuple[List[str], List[int]]:
    """Load existing wild dataset for comparison."""
    print("Loading wild dataset...")
    try:
        # Try to find existing results
        results_path = Path(__file__).parent.parent / "data" / "results" / "sfi_wild_10k_results.csv"
        if not results_path.exists():
            print(f"  Wild dataset not found at {results_path}")
            return [], []

        df = pd.read_csv(results_path)
        texts = df['text'].tolist()
        labels = df['is_jailbreak'].astype(int).tolist()

        print(f"  Loaded {len(texts)} examples ({sum(labels)} malicious, {len(labels)-sum(labels)} safe)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading wild dataset: {e}")
        return [], []


def compute_rsfi_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    k: int = 20,
    apply_whitening: bool = True
) -> np.ndarray:
    """
    Compute RSFI scores using SVD-based subspace projection.

    Args:
        embeddings: All embeddings (N, d)
        ref_indices: Indices of reference (malicious) examples
        k: Number of SVD components
        apply_whitening: Whether to apply ZCA whitening

    Returns:
        scores: RSFI scores for all examples (higher = more malicious)
    """
    # Split reference and test
    ref_emb = embeddings[ref_indices]

    # Apply whitening if requested
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

    # Project onto top-k subspace
    U_k = U[:, :k]

    # Compute projection magnitude (RSFI score)
    proj = embeddings_wh @ U_k  # (N, k)
    scores = np.linalg.norm(proj, axis=1)  # (N,)

    return scores


def compute_naive_cosine_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray
) -> np.ndarray:
    """
    Compute naive cosine similarity to mean of reference set.
    """
    ref_emb = embeddings[ref_indices]
    ref_mean = np.mean(ref_emb, axis=0, keepdims=True)
    ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-15)

    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-15)
    scores = embeddings_norm @ ref_mean.T
    return scores.flatten()


def evaluate_dataset(
    texts: List[str],
    labels: List[int],
    model: SentenceTransformer,
    dataset_name: str,
    n_ref: int = 200,
    n_val: int = 200,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Run honest evaluation protocol on a single dataset.

    Returns DataFrame with one row per (seed, method) combination.
    """
    if len(texts) == 0:
        print(f"  Skipping {dataset_name} (no data)")
        return pd.DataFrame()

    print(f"\nEvaluating {dataset_name}...")
    print(f"  Total examples: {len(texts)} ({sum(labels)} malicious)")

    # Encode all texts
    print("  Encoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    labels = np.array(labels)

    results = []

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}")

        # Split: ref (malicious only), val, test
        malicious_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]

        if len(malicious_idx) < n_ref + 50:
            print(f"    WARNING: Not enough malicious examples ({len(malicious_idx)} < {n_ref + 50})")
            continue

        if len(safe_idx) < n_val + 100:
            print(f"    WARNING: Not enough safe examples ({len(safe_idx)} < {n_val + 100})")
            continue

        # Reference set: n_ref malicious examples
        np.random.seed(seed)
        ref_idx = np.random.choice(malicious_idx, size=n_ref, replace=False)
        remaining_mal_idx = np.setdiff1d(malicious_idx, ref_idx)

        # Validation: n_val examples (balanced)
        val_mal_idx = np.random.choice(remaining_mal_idx, size=n_val//2, replace=False)
        val_safe_idx = np.random.choice(safe_idx, size=n_val//2, replace=False)
        val_idx = np.concatenate([val_mal_idx, val_safe_idx])

        # Test: everything else
        test_idx = np.setdiff1d(np.arange(len(texts)), np.concatenate([ref_idx, val_idx]))

        y_val = labels[val_idx]
        y_test = labels[test_idx]

        print(f"    Splits: ref={len(ref_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Method 1: RSFI-SVD (with whitening)
        scores_rsfi_val = compute_rsfi_scores(embeddings[val_idx], np.arange(n_ref), k=20, apply_whitening=True)
        scores_rsfi_test = compute_rsfi_scores(
            np.vstack([embeddings[ref_idx], embeddings[test_idx]]),
            np.arange(n_ref),
            k=20,
            apply_whitening=True
        )[n_ref:]

        auc_rsfi = roc_auc_score(y_test, scores_rsfi_test)
        pr_auc_rsfi = average_precision_score(y_test, scores_rsfi_test)

        results.append({
            'dataset': dataset_name,
            'seed': seed,
            'method': 'RSFI-SVD',
            'n_ref': n_ref,
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'roc_auc': auc_rsfi,
            'pr_auc': pr_auc_rsfi
        })

        # Method 2: Naive cosine
        scores_cosine_test = compute_naive_cosine_scores(
            np.vstack([embeddings[ref_idx], embeddings[test_idx]]),
            np.arange(n_ref)
        )[n_ref:]

        auc_cosine = roc_auc_score(y_test, scores_cosine_test)
        pr_auc_cosine = average_precision_score(y_test, scores_cosine_test)

        results.append({
            'dataset': dataset_name,
            'seed': seed,
            'method': 'naive_cosine',
            'n_ref': n_ref,
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'roc_auc': auc_cosine,
            'pr_auc': pr_auc_cosine
        })

        # Method 3: LogReg
        logreg = LogisticRegression(max_iter=1000, random_state=seed)

        # Train on ref + val
        train_idx_local = np.concatenate([np.arange(n_ref), n_ref + np.arange(len(val_idx))])
        test_idx_local = n_ref + len(val_idx) + np.arange(len(test_idx))

        X_all = np.vstack([embeddings[ref_idx], embeddings[val_idx], embeddings[test_idx]])
        y_all = np.concatenate([labels[ref_idx], y_val, y_test])

        logreg.fit(X_all[train_idx_local], y_all[train_idx_local])
        scores_logreg_test = logreg.decision_function(X_all[test_idx_local])

        auc_logreg = roc_auc_score(y_test, scores_logreg_test)
        pr_auc_logreg = average_precision_score(y_test, scores_logreg_test)

        results.append({
            'dataset': dataset_name,
            'seed': seed,
            'method': 'LogReg',
            'n_ref': n_ref + len(val_idx),
            'n_val': 0,
            'n_test': len(test_idx),
            'roc_auc': auc_logreg,
            'pr_auc': pr_auc_logreg
        })

        print(f"    RSFI: {auc_rsfi:.4f}, Cosine: {auc_cosine:.4f}, LogReg: {auc_logreg:.4f}")

    return pd.DataFrame(results)


def main():
    """Run E2 experiment across all datasets."""
    print("="*80)
    print("E2: HOMOGENEOUS DATASETS EXPERIMENT")
    print("="*80)

    # Configuration
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    N_REF = 200
    N_VAL = 200
    N_SEEDS = 5

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  N_ref: {N_REF}")
    print(f"  N_val: {N_VAL}")
    print(f"  N_seeds: {N_SEEDS}")

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Load datasets
    datasets = []

    toxicchat_texts, toxicchat_labels = load_toxicchat()
    if len(toxicchat_texts) > 0:
        datasets.append(("ToxicChat", toxicchat_texts, toxicchat_labels))

    xstest_texts, xstest_labels = load_xstest()
    if len(xstest_texts) > 0:
        datasets.append(("XSTest", xstest_texts, xstest_labels))

    wild_texts, wild_labels = load_wild_dataset()
    if len(wild_texts) > 0:
        datasets.append(("Wild", wild_texts, wild_labels))

    if len(datasets) == 0:
        print("\nERROR: No datasets loaded. Cannot run experiment.")
        return

    # Run evaluation on each dataset
    all_results = []

    for dataset_name, texts, labels in datasets:
        df = evaluate_dataset(
            texts, labels, model, dataset_name,
            n_ref=N_REF, n_val=N_VAL, n_seeds=N_SEEDS
        )
        if len(df) > 0:
            all_results.append(df)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "E2_homogeneous_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    summary = results_df.groupby(['dataset', 'method'])[['roc_auc', 'pr_auc']].agg(['mean', 'std'])
    print(summary.to_string())

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("Expected: All methods drop on homogeneous datasets (ToxicChat, XSTest).")
    print("Key question: Does RSFI drop LESS than naive cosine?")
    print("If yes → SVD contribution confirmed even on hard data.")
    print("If no → Method limited to stylistically diverse datasets (honest finding).")


if __name__ == "__main__":
    main()
