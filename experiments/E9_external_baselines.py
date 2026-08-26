"""
E9_external_baselines.py
================================================================================
D5 fix: honest head-to-head against EXTERNAL published classifiers on the SAME
leakage-free splits as E2d/E8 (same loaders, same budgets, 5 seeds).

External baselines (zero-shot, no calibration on our data):
  - protectai/deberta-v3-base-prompt-injection-v2
      (DeBERTa-v3-base fine-tuned for prompt injection / jailbreak detection;
       score = softmax P(INJECTION))
  - unitary/toxic-bert
      (BERT fine-tuned on Jigsaw toxicity; score = max toxic-label probability;
       domain-mismatched for jailbreaks, included deliberately to show it)

In-repo reference methods (mpnet embeddings from cache):
  - B1_raw, B1b_SigmaT_wh (discriminant direction, as in E8)

Methodological notes:
  - External classifiers do not use the reference set at all, so their scores
    are computed ONCE per dataset over all texts; per-seed rows just evaluate
    them on that seed's test subset. This is the fair protocol: the external
    model sees exactly the same test items as our methods.
  - Domain mismatch is expected and reported, not hidden: prompt-injection /
    toxicity classifiers are not jailbreak classifiers. The comparison answers
    "is a lightweight discriminant filter competitive with an off-the-shelf
    published classifier", not "which is the best possible detector".

Output: data/results/E9_external_baselines.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_wild,
    load_xstest,
    get_embeddings,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener,
    PooledWithinClassWhitening,
    score_discriminant,
    l2norm,
)

N_SEEDS = 5


def l2norm_local(x):
    return l2norm(x)


class ExternalClassifier:
    """Zero-shot transformer classifier wrapper."""

    def __init__(self, model_id, score_mode="softmax_class1", label_index=1):
        self.model_id = model_id
        self.score_mode = score_mode
        self.label_index = label_index
        self.available = False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            print(f"  Loading {model_id} ...")
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            print(f"    Loaded on {self.device}; id2label={self.model.config.id2label}")
        except Exception as e:
            print(f"    WARNING: could not load {model_id}: {e}")

    def score_all(self, texts, batch_size=32, max_length=512):
        """Return anomaly score per text (higher = more likely malicious)."""
        if not self.available:
            return None
        torch = self.torch
        scores = []
        t0 = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = [str(t)[:4000] for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if self.score_mode == "softmax_class1":
                scores.extend(probs[:, self.label_index])
            elif self.score_mode == "max_prob":
                scores.extend(probs.max(axis=1))
        dt = time.perf_counter() - t0
        print(f"    Scored {len(texts)} texts in {dt:.1f}s "
              f"({dt / len(texts) * 1000:.2f} ms/text)")
        return np.array(scores)


def run():
    print("=" * 80)
    print("E9: EXTERNAL BASELINES ON SHARED SPLITS (D5 fix)")
    print("=" * 80)

    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }

    externals = [
        ExternalClassifier(
            "protectai/deberta-v3-base-prompt-injection-v2",
            score_mode="softmax_class1", label_index=1
        ),
        ExternalClassifier("unitary/toxic-bert", score_mode="max_prob"),
    ]

    embedder = "sentence-transformers/all-mpnet-base-v2"
    rows = []

    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal, n_safe = int(labels.sum()), len(labels) - int(labels.sum())
        print(f"\n### {d_name}: {len(texts)} items ({n_mal} mal / {n_safe} safe)")

        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = n_ref_safe = 200

        # --- external scores: computed once, independent of seed ---
        ext_scores = {}
        for ext in externals:
            if ext.available:
                s = ext.score_all(texts)
                if s is not None:
                    ext_scores[ext.model_id] = s

        # --- our methods: mpnet embeddings from cache ---
        embeddings = get_embeddings(texts, d_name, embedder)
        dim = embeddings.shape[1]

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

            base = {"dataset": d_name, "seed": seed, "n_test": len(test_idx),
                    "n_ref_mal": n_ref_mal, "n_ref_safe": n_ref_safe}

            # external rows (scores sliced to this seed's test set)
            for model_id, s_all in ext_scores.items():
                short = model_id.split("/")[-1]
                rows.append(dict(base, method=f"EXT:{short}",
                                 roc_auc=roc_auc_score(y_test, s_all[test_idx]),
                                 pr_auc=average_precision_score(y_test, s_all[test_idx])))

            # our B1 / B1b (identical protocol to E8)
            emb_test = embeddings[test_idx]
            ref_mal_raw = embeddings[ref_mal_idx]
            ref_safe_raw = embeddings[ref_safe_idx]
            wh_t = fit_sigma_t_whitener(np.vstack([ref_mal_raw, ref_safe_raw]), dim)
            test_t = wh_t.transform(emb_test)
            mal_t = wh_t.transform(ref_mal_raw)
            safe_t = wh_t.transform(ref_safe_raw)

            ours = {
                "OURS:B1_raw": score_discriminant(emb_test, ref_mal_raw, ref_safe_raw),
                "OURS:B1b_SigmaT_wh": score_discriminant(test_t, mal_t, safe_t),
            }
            for m, sc in ours.items():
                rows.append(dict(base, method=m,
                                 roc_auc=roc_auc_score(y_test, sc),
                                 pr_auc=average_precision_score(y_test, sc)))

    out_dir = Path(__file__).parent.parent / "data" / "results"
    df = pd.DataFrame(rows)
    out_csv = out_dir / "E9_external_baselines.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY (ROC-AUC mean +- std over 5 seeds) ===")
    summary = df.groupby(["dataset", "method"]).roc_auc.agg(["mean", "std"]).round(4)
    print(summary.to_string())
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    run()