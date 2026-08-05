"""
E7 — ВНЕШНИЕ БЕЙЗЛАЙНЫ на общих данных (последний P0-блок).

Без этой таблицы статью не примут: метод обязан сравниваться с тем, что уже
опубликовано и развёрнуто в индустрии, на ИДЕНТИЧНЫХ сплитах и датасетах.

Бейзлайны:
  1. Meta Prompt-Guard-86M (zero-shot классификатор, индустриальный стандарт)
     ВНИМАНИЕ: репозиторий gated — нужно принять лицензию на HF и задать HF_TOKEN.
       set HF_TOKEN=hf_...   (Windows)  /  export HF_TOKEN=hf_...  (bash)
     Если недоступен — скрипт честно пропустит его и напишет почему.
  2. ProtectAI deberta-v3-base-prompt-injection-v2 (open, не gated).
  3. Кодбук ИТМО (k-NN cosine): score = max cosine к эмбеддингам ref-атак
     (ровно метод Alanova et al. 2026, arXiv:2604.25716).
  4. RSFI FULL(ZCA+SVD) — для head-to-head в той же таблице.
  5. LOGREG(ref) — контроль равного бюджета.

Датасеты и сплиты: как в E2/E3 (wild + ToxicChat x2 + XSTest, 5 сидов,
ref/val/test, эмбеддинги из кэша E2). Метрики: AUC, AUPRC, TPR@FPR<=1%
(порог по валидации; у внешних классификаторов «валидация» используется
только для калибровки порога — они сами zero-shot).

Запуск:
  python honest_eval_e7.py --models all-mpnet-base-v2 BAAI/bge-base-en-v1.5
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score

from honest_eval_e2 import load_datasets, get_embeddings
from honest_eval_e3 import operating_point, get_scores, TARGET_FPRS, SEEDS

SCORE_CACHE = Path("emb_cache")


def hf_classifier_scores(hf_name, ds_name, texts, mode):
    """Zero-shot скоры HF-классификатора. mode: 'prompt_guard' | 'deberta_pi'."""
    f = SCORE_CACHE / f"e7_{ds_name}_{hf_name.replace('/', '_')}.npy"
    if f.exists():
        return np.load(f)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"[HF] {hf_name} on {ds_name} ...")
    tok = AutoTokenizer.from_pretrained(hf_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(hf_name)
    mdl.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(texts[i:i+32], truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(dev)
            probs = torch.softmax(mdl(**batch).logits, dim=-1).cpu().numpy()
            if mode == "prompt_guard":
                # метки: 0 benign, 1 injection, 2 jailbreak -> скор = P(inj)+P(jail)
                scores.extend(probs[:, 1] + probs[:, 2] if probs.shape[1] >= 3 else probs[:, -1])
            else:
                # deberta-prompt-injection: последний класс = INJECTION
                scores.extend(probs[:, -1])
    scores = np.asarray(scores)
    np.save(f, scores)
    return scores


def codebook_scores(E, ref_mal, idx, topk=1):
    """Кодбук ИТМО: max (или mean-topk) cosine к ref-атакам."""
    from rsfi.geometry import RiemannianSphere
    En = RiemannianSphere.normalize(E)
    cb = En[ref_mal]
    sims = En[idx] @ cb.T
    if topk == 1:
        return sims.max(axis=1)
    return np.sort(sims, axis=1)[:, -topk:].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"])
    args = ap.parse_args()

    df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
    datasets = {"wild_2000": (df["text"].tolist(),
                              (df["scenario_type"] == "MALICIOUS").astype(int).values)}
    datasets.update(load_datasets())

    rows = []
    for ds_name, (texts, y) in datasets.items():
        n_ref = 50 if ds_name == "xstest" else (150 if ds_name.startswith("toxicchat") else 200)

        # --- внешние zero-shot классификаторы (один скоринг на датасет) ---
        ext = {}
        for hf_name, mode in [("meta-llama/Prompt-Guard-86M", "prompt_guard"),
                              ("protectai/deberta-v3-base-prompt-injection-v2", "deberta_pi")]:
            try:
                ext[hf_name] = hf_classifier_scores(hf_name, ds_name, texts, mode)
            except Exception as e:
                print(f"  [skip] {hf_name}: {str(e)[:150]}")

        for model_name in args.models:
            cache_name = ds_name if ds_name != "wild_2000" else "wild2000"
            E = get_embeddings(model_name, cache_name, texts)
            print(f"\n===== E7 | {ds_name} | {model_name} =====")

            methods = ["RSFI FULL", "LOGREG(ref)", "CODEBOOK-1NN", "CODEBOOK-5NN"] + list(ext.keys())
            agg = {m: {"auc": [], "auprc": [], "tpr1": []} for m in methods}

            for seed in range(SEEDS):
                rng = np.random.RandomState(seed)
                mal = rng.permutation(np.where(y == 1)[0])
                safe = rng.permutation(np.where(y == 0)[0])
                n_val = min(200, len(mal) - n_ref, len(safe) - n_ref)
                if n_val < 20:
                    break
                ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
                val_idx = np.concatenate([mal[n_ref:n_ref+n_val], safe[n_ref:n_ref+n_val]])
                test_idx = np.concatenate([mal[n_ref+n_val:], safe[n_ref+n_val:]])
                y_val, y_test = y[val_idx], y[test_idx]

                val_sc, test_sc = get_scores("FULL(ZCA+SVD)", E, ref_mal, ref_safe, val_idx, test_idx, y_val)
                for m, (v_sc, t_sc) in {
                    "RSFI FULL": (val_sc, test_sc),
                    "LOGREG(ref)": get_scores("LOGREG(ref)", E, ref_mal, ref_safe, val_idx, test_idx, y_val),
                    "CODEBOOK-1NN": (codebook_scores(E, ref_mal, val_idx, 1), codebook_scores(E, ref_mal, test_idx, 1)),
                    "CODEBOOK-5NN": (codebook_scores(E, ref_mal, val_idx, 5), codebook_scores(E, ref_mal, test_idx, 5)),
                }.items():
                    agg[m]["auc"].append(roc_auc_score(y_test, t_sc))
                    agg[m]["auprc"].append(average_precision_score(y_test, t_sc))
                    agg[m]["tpr1"].append(operating_point(v_sc, y_val, t_sc, y_test, 0.01)[0])

                for hf_name, sc_all in ext.items():
                    t_sc, v_sc = sc_all[test_idx], sc_all[val_idx]
                    agg[hf_name]["auc"].append(roc_auc_score(y_test, t_sc))
                    agg[hf_name]["auprc"].append(average_precision_score(y_test, t_sc))
                    agg[hf_name]["tpr1"].append(operating_point(v_sc, y_val, t_sc, y_test, 0.01)[0])

            for m, v in agg.items():
                if not v["auc"]:
                    continue
                print(f"  {m:42s}: AUC {np.mean(v['auc']):.4f}±{np.std(v['auc']):.3f} | "
                      f"AUPRC {np.mean(v['auprc']):.4f} | TPR@FPR1% {np.mean(v['tpr1']):.3f}")
                rows.append({"dataset": ds_name, "embed_model": model_name, "method": m,
                             "auc_mean": np.mean(v["auc"]), "auc_std": np.std(v["auc"]),
                             "auprc_mean": np.mean(v["auprc"]),
                             "tpr_at_fpr1_mean": np.mean(v["tpr1"]), "tpr_at_fpr1_std": np.std(v["tpr1"])})

    out = pd.DataFrame(rows)
    out.to_csv("e7_external_baselines_results.csv", index=False)
    print("\nSaved: e7_external_baselines_results.csv")


if __name__ == "__main__":
    main()
