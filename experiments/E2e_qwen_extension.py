"""
E2e_qwen_extension.py
================================================================================
Optional-audit item: extend the reproducible per-seed protocol to
Qwen/Qwen3-Embedding-8B (d=4096), which previously existed only as a legacy
text log (audit defect D7) and as missing cells (A3 / C1b dashes in Table 2,
XSTest row absent, no Sigma_W block).

This script:
  1. Encodes ToxicChat and XSTest with Qwen3-Embedding-8B on CUDA and caches
     embeddings under emb_cache/{dataset}_Qwen_Qwen3-Embedding-8B.npy
     (Wild cache already exists; it is REUSED verbatim so that the Wild rows
     reproduce the legacy numbers on identical data).
  2. Runs the E2d method battery (A1/A1b/A2/A3/B1/B1b/C1/C1b + DeLong pairs)
     for all three datasets, 5 seeds, leakage-free splits.
  3. Runs the E8 Sigma_W block (B1_raw / B1b_SigmaT_wh / B1w_SigmaW_wh +
     DeLong) for all three datasets.

Embedding definition (documented choice): last non-padding token hidden state
of the raw Qwen3 backbone, stored UNNORMALIZED float32 ("raw" semantics of
every other cache in this repo). All repo methods L2-normalize internally,
except C1 LogReg / whitening paths whose absolute values are invariant to a
global rescale of both calibration and test vectors together.

Outputs:
  data/results/E2q_qwen_multidataset.csv   (schema = E2d)
  data/results/E8q_qwen_sigma_w.csv        (schema = E8_sigma_w.csv group B)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_wild,
    load_xstest,
    l2norm,
    fit_whitener,
    score_mean_direction,
    score_svd_subspace,
    score_discriminant_mean,
    delong_test,
    N_SEEDS,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener,
    PooledWithinClassWhitening,
    score_discriminant,
)

QWEN_ID = "Qwen/Qwen3-Embedding-8B"
EMB_CACHE = Path(__file__).parent.parent / "emb_cache"
MAX_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _encode_qwen_manual(texts):
    """Fallback encoder: last non-padding token hidden state, on CPU or GPU."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModel.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "10GB", "cpu": "20GB"} if DEVICE == "cuda" else None,
    )
    model.eval()
    print(f"  [ManualQwen] loaded via device_map=auto, dtype={next(model.parameters()).dtype}",
          flush=True)
    out = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
            if DEVICE == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            hs = model(**enc).last_hidden_state          # (B, L, D)
            last_idx = (enc["attention_mask"].sum(dim=1) - 1).to(hs.device)  # (B,)
            vecs = hs[torch.arange(hs.size(0), device=hs.device), last_idx]
            out.append(vecs.detach().float().cpu().numpy())
            if (i // bs) % 20 == 0:
                print(f"    manual encode {i + len(batch)}/{len(texts)}",
                      flush=True)
    import gc
    del model, tok
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out)


def get_qwen_embeddings(texts, dataset_name):
    cache_file = EMB_CACHE / f"{dataset_name}_Qwen_Qwen3-Embedding-8B.npy"
    # Match pre-existing cache naming if applicable (E2d protocol)
    if dataset_name == "Wild":
        legacy = EMB_CACHE / "Qwen_Qwen3-Embedding-8B.npy"
        if legacy.exists():
            cache_file = legacy
    if cache_file.exists():
        emb = np.load(cache_file)
        assert emb.shape[0] == len(texts), \
            f"cache {cache_file.name} rows {emb.shape[0]} != {len(texts)}"
        print(f"  [Cache hit] {cache_file.name} shape={emb.shape}", flush=True)
        return emb

    print(f"  [Encoding] {len(texts)} texts with {QWEN_ID}...",
          flush=True)
    emb = _encode_qwen_manual(texts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    print(f"  [Cached] {cache_file.name} shape={emb.shape}", flush=True)
    return emb


# ---------------------------------------------------------------------------
# Benchmarks (protocols replicated verbatim from E2d / E8)
# ---------------------------------------------------------------------------

def budget_for(n_mal, n_safe):
    if n_mal < 250 or n_safe < 250:
        return max(10, n_mal // 3), max(10, n_safe // 3)
    return 200, 200


def e2d_battery(d_name, texts, labels, embeddings):
    labels = np.array(labels)
    n_mal = int(labels.sum())
    n_safe = len(labels) - n_mal
    dim = embeddings.shape[1]
    n_ref_mal, n_ref_safe = budget_for(n_mal, n_safe)
    model_short = "Qwen3-Embedding-8B"
    print(f"\n[E2q battery] {d_name}: {n_mal}/{n_safe}, "
          f"budget {n_ref_mal}/{n_ref_safe}", flush=True)

    mal_idx_arr = np.where(labels == 1)[0]
    safe_idx_arr = np.where(labels == 0)[0]
    rows_res, rows_delong = [], []

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        ref_mal_idx = np.random.choice(mal_idx_arr, size=n_ref_mal, replace=False)
        ref_safe_idx = np.random.choice(safe_idx_arr, size=n_ref_safe, replace=False)
        test_idx = np.concatenate([
            np.setdiff1d(mal_idx_arr, ref_mal_idx),
            np.setdiff1d(safe_idx_arr, ref_safe_idx),
        ])
        y_test = labels[test_idx]

        emb_test = embeddings[test_idx]
        ref_mal_raw = embeddings[ref_mal_idx]
        ref_safe_raw = embeddings[ref_safe_idx]
        ref_comb = np.vstack([ref_mal_raw, ref_safe_raw])

        wh = fit_whitener(ref_comb, dim=dim)
        emb_test_wh = wh.transform(emb_test)
        ref_mal_wh = wh.transform(ref_mal_raw)
        ref_safe_wh = wh.transform(ref_safe_raw)

        k = min(20, n_ref_mal)
        scores = {
            "A1_naive_cosine_raw": score_mean_direction(emb_test, ref_mal_raw),
            "A1b_cosine_whitened": score_mean_direction(emb_test_wh, ref_mal_wh),
            "A2_rsfi_svd_raw_k20": score_svd_subspace(emb_test, ref_mal_raw, k=k),
            "A3_rsfi_svd_whitened_k20": score_svd_subspace(emb_test_wh, ref_mal_wh, k=k),
            "B1_discriminant_mean_raw": score_discriminant_mean(
                emb_test, ref_mal_raw, ref_safe_raw),
            "B1b_discriminant_mean_whitened": score_discriminant_mean(
                emb_test_wh, ref_mal_wh, ref_safe_wh),
        }
        y_train = np.concatenate([np.ones(n_ref_mal), np.zeros(n_ref_safe)])
        lr_raw = LogisticRegression(max_iter=1000, random_state=seed)
        lr_raw.fit(ref_comb, y_train)
        scores["C1_logreg_raw"] = lr_raw.decision_function(emb_test)

        lr_wh = LogisticRegression(max_iter=1000, random_state=seed)
        lr_wh.fit(np.vstack([ref_mal_wh, ref_safe_wh]), y_train)
        scores["C1b_logreg_whitened"] = lr_wh.decision_function(emb_test_wh)

        for m_name, sc in scores.items():
            rows_res.append(dict(
                dataset=d_name, model=model_short, dim=dim, seed=seed,
                method=m_name,
                roc_auc=float(roc_auc_score(y_test, sc)),
                pr_auc=float(average_precision_score(y_test, sc)),
                n_test=len(test_idx),
            ))

        for m1, m2, tag in [
            ("B1b_discriminant_mean_whitened", "C1b_logreg_whitened", "B1b_vs_C1b"),
            ("B1_discriminant_mean_raw", "A1_naive_cosine_raw", "B1_vs_A1"),
            ("B1_discriminant_mean_raw", "C1_logreg_raw", "B1_vs_C1"),
            ("A3_rsfi_svd_whitened_k20", "A1_naive_cosine_raw", "A3_vs_A1"),
            ("B1b_discriminant_mean_whitened", "B1_discriminant_mean_raw", "B1b_vs_B1"),
            ("B1b_discriminant_mean_whitened", "A1b_cosine_whitened", "B1b_vs_A1b"),
        ]:
            auc1 = roc_auc_score(y_test, scores[m1])
            auc2 = roc_auc_score(y_test, scores[m2])
            rows_delong.append(dict(
                dataset=d_name, model=model_short, seed=seed, pair=tag,
                method_1=m1, auc_1=float(auc1),
                method_2=m2, auc_2=float(auc2),
                auc_diff=float(auc1 - auc2),
                p_value=delong_test(y_test, scores[m1], scores[m2]),
            ))
        print(f"  seed{seed} done "
              f"(B1={roc_auc_score(y_test, scores['B1_discriminant_mean_raw']):.4f})", flush=True)

    return pd.DataFrame(rows_res), pd.DataFrame(rows_delong)

def sigma_w_block(d_name, texts, labels, embeddings):
    """E8 group-B replica for Qwen3-8B (B1_raw / B1b / B1w + DeLong)."""
    labels = np.array(labels)
    n_mal = int(labels.sum())
    n_safe = len(labels) - n_mal
    dim = embeddings.shape[1]
    n_ref_mal, n_ref_safe = budget_for(n_mal, n_safe)
    model_short = "Qwen3-Embedding-8B"
    print(f"\n[E8q block] {d_name}: budget {n_ref_mal}/{n_ref_safe}",
          flush=True)

    mal_idx_arr = np.where(labels == 1)[0]
    safe_idx_arr = np.where(labels == 0)[0]
    rows_b, rows_d = [], []

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        ref_mal_idx = np.random.choice(mal_idx_arr, size=n_ref_mal, replace=False)
        ref_safe_idx = np.random.choice(safe_idx_arr, size=n_ref_safe, replace=False)
        test_idx = np.concatenate([
            np.setdiff1d(mal_idx_arr, ref_mal_idx),
            np.setdiff1d(safe_idx_arr, ref_safe_idx),
        ])
        y_test = labels[test_idx]

        emb_test = embeddings[test_idx]
        ref_mal_raw = embeddings[ref_mal_idx]
        ref_safe_raw = embeddings[ref_safe_idx]
        wh_t = fit_sigma_t_whitener(np.vstack([ref_mal_raw, ref_safe_raw]), dim)
        wh_w = PooledWithinClassWhitening(dim).fit(ref_mal_raw, ref_safe_raw)

        test_t = wh_t.transform(emb_test)
        mal_t = wh_t.transform(ref_mal_raw)
        safe_t = wh_t.transform(ref_safe_raw)

        test_w = wh_w.transform(emb_test)
        mal_w = wh_w.transform(ref_mal_raw)
        safe_w = wh_w.transform(ref_safe_raw)

        scores_b = {
            "B1_raw": score_discriminant(emb_test, ref_mal_raw, ref_safe_raw),
            "B1b_SigmaT_wh": score_discriminant(test_t, mal_t, safe_t),
            "B1w_SigmaW_wh": score_discriminant(test_w, mal_w, safe_w),
        }

        base = dict(dataset=d_name, model=model_short, dim=dim, seed=seed,
                    n_ref_mal=n_ref_mal, n_ref_safe=n_ref_safe,
                    n_test=len(test_idx))
        for m, sc in scores_b.items():
            rows_b.append(dict(base, method=m,
                                roc_auc=float(roc_auc_score(y_test, sc)),
                                pr_auc=float(average_precision_score(y_test, sc))))

        for m1, m2, tag in [
            ("B1w_SigmaW_wh", "B1b_SigmaT_wh", "B1w_vs_B1b"),
            ("B1b_SigmaT_wh", "B1_raw", "B1b_vs_B1"),
            ("B1w_SigmaW_wh", "B1_raw", "B1w_vs_B1"),
        ]:
            auc1 = roc_auc_score(y_test, scores_b[m1])
            auc2 = roc_auc_score(y_test, scores_b[m2])
            rows_d.append(dict(
                base, pair=tag, method_1=m1, method_2=m2,
                auc_diff=float(auc1 - auc2),
                p_value=delong_test(y_test, scores_b[m1], scores_b[m2]),
            ))
        print(f"  seed{seed} done "
              f"(B1w={roc_auc_score(y_test, scores_b['B1w_SigmaW_wh']):.4f})", flush=True)

    return pd.DataFrame(rows_b), pd.DataFrame(rows_d)


def main():
    t0 = time.time()
    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }
    print("=" * 80, flush=True)
    print("E2e: Qwen3-Embedding-8B per-seed extension (E2d battery + E8 block)",
          flush=True)
    print(f"Device: {DEVICE}", flush=True)

    embs = {}
    for d_name, (texts, _labels) in datasets.items():
        embs[d_name] = get_qwen_embeddings(texts, d_name)

    res_frames, delong_frames = [], []
    sw_frames, swd_frames = [], []
    for d_name, (texts, labels) in datasets.items():
        r, dl = e2d_battery(d_name, texts, labels, embs[d_name])
        res_frames.append(r)
        delong_frames.append(dl)

        b, d = sigma_w_block(d_name, texts, labels, embs[d_name])
        sw_frames.append(b)
        swd_frames.append(d)

    out_dir = Path(__file__).parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_res = pd.concat(res_frames, ignore_index=True)
    df_delong = pd.concat(delong_frames, ignore_index=True)
    df_sw = pd.concat(sw_frames, ignore_index=True)
    df_swd = pd.concat(swd_frames, ignore_index=True)

    df_res.to_csv(out_dir / "E2q_qwen_multidataset.csv", index=False)
    df_delong.to_csv(out_dir / "E2q_qwen_delong_tests.csv", index=False)
    df_sw.to_csv(out_dir / "E8q_qwen_sigma_w.csv", index=False)
    df_swd.to_csv(out_dir / "E8q_qwen_delong_tests.csv", index=False)
    print(f"\nSaved 4 CSVs to {out_dir} "
          f"(total {time.time() - t0:.0f}s)", flush=True)

    piv = df_res.pivot_table(index=["dataset"], columns="method",
                             values="roc_auc")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200):
        print("\n=== E2q ROC-AUC means over seeds ===")
        print(piv.round(4).to_string())
        piv_sw = df_sw.pivot_table(index=["dataset"], columns="method",
                                   values="roc_auc")
        print("\n=== E8q (Sigma_W) ROC-AUC means over seeds ===")
        print(piv_sw.round(4).to_string())

    legacy = {("Wild", "A1_naive_cosine_raw"): 0.7982,
              ("Wild", "A2_rsfi_svd_raw_k20"): 0.8012,
              ("Wild", "B1_discriminant_mean_raw"): 0.8747,
              ("Wild", "C1_logreg_raw"): 0.8846}
    print("\n=== Reconciliation vs legacy log (Wild) ===")
    for (ds, m), exp in legacy.items():
        got = df_res[(df_res.dataset == ds)
                     & (df_res.method == m)].roc_auc.mean()
        print(f"{ds}/{m}: this run {got:.4f} vs legacy {exp:.4f} "
              f"(delta {got - exp:+.4f})", flush=True)


if __name__ == "__main__":
    main()



