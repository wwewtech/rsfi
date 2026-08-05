"""
E2 — ОДНОРОДНЫЕ ДАТАСЕТЫ: ToxicChat-0124 и XSTest-v2 (решающий тест).

Почему это P0: текущий датасет проекта (TrustAIRLab vs WildChat) стилистически
неоднороден — длинные шаблонные джейлбрейки против обычных промптов различимы
даже по длине. ToxicChat и XSTest — стандарт де-факто (GradSafe, ACL 2024):
реальные юзеры (LMSYS) и контрастные пары («kill a process» vs «kill a person»).

Датасеты (скачаются с HF автоматически):
  1. ToxicChat-0124 (lmsys/toxic-chat): реальные промпты юзеров.
     Две постановки:
       a) toxicity=1 -> MALICIOUS (как у GradSafe)
       b) jailbreaking=1 -> MALICIOUS (строго джейлбрейки)
  2. XSTest-v2 (paulrottger/xstest): 250 safe / 200 unsafe контрастных пар.
     Тест на over-refusal: FPR на safe с «опасными» словами.

Протокол: честный (как v2-v6): whitening только на ref-safe, SVD на ref-атаках,
k по валидации, бейзлайны при равном бюджете (LogReg на ref), 5 сидов.
Метрики: ROC-AUC + AUPRC (стандарт GradSafe).

Модели по умолчанию: mpnet + bge-base (CPU-доступно).
Qwen3-Embedding-8B — добавить флагом (GPU):
  python honest_eval_e2.py --models all-mpnet-base-v2 BAAI/bge-base-en-v1.5 Qwen/Qwen3-Embedding-8B
"""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

K_LIST = [1, 5, 10, 20, 30, 40]
SEEDS = 5
CACHE = Path("emb_cache"); CACHE.mkdir(exist_ok=True)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def svd_filter_auc(E_space, ref_mal, ref_safe, val_idx, test_idx, y_val, y_test):
    from rsfi.filter import MultiDimensionalRSFIFilter
    from rsfi.geometry import RiemannianSphere
    S = RiemannianSphere.normalize(E_space[ref_safe].mean(axis=0, keepdims=True))[0]
    tang = np.array([RiemannianSphere.log_map(S, v) for v in E_space[ref_mal]])
    _, _, Vh = np.linalg.svd(tang, full_matrices=False)
    aucs = {}
    for k in [kk for kk in K_LIST if kk <= Vh.shape[0]]:
        flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
        aucs[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in E_space[val_idx]])
    bk = max(aucs, key=aucs.get)
    flt = MultiDimensionalRSFIFilter(S, Vh[:bk], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
    sc = np.array([-flt.evaluate(v)["rsfi"] for v in E_space[test_idx]])
    return sc, bk


def load_datasets():
    from datasets import load_dataset
    out = {}
    print("[DATA] ToxicChat-0124 ...")
    tc = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    texts = [t if isinstance(t, str) and t.strip() else "" for t in tc["user_input"]]
    out["toxicchat_toxicity"] = (texts, np.array(tc["toxicity"], dtype=int))
    out["toxicchat_jailbreak"] = (texts, np.array(tc["jailbreaking"], dtype=int))
    print("[DATA] XSTest-v2 ...")
    # оригинальный paulrottger/xstest удалён с HF; используем официальную копию v2.
    # Метки: unsafe = типы 'contrast_*' (200), safe = остальные 10 типов (250) —
    # соответствует официальной разметке XSTest-v2 (250 safe / 200 unsafe).
    xs = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    out["xstest"] = (list(xs["prompt"]),
                     np.array([1 if t.startswith("contrast_") else 0 for t in xs["type"]], dtype=int))
    for k, (t, y) in out.items():
        print(f"  {k}: N={len(t)}, malicious={y.sum()}, safe={len(y)-y.sum()}")
    return out


def get_embeddings(model_name, ds_name, texts):
    f = CACHE / f"e2_{ds_name}_{model_name.replace('/', '_')}.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENCODE] {ds_name} with {model_name} on {dev} ...")
    t0 = time.time()
    m = SentenceTransformer(model_name, device=dev,
                            model_kwargs={"dtype": "bfloat16"} if dev == "cuda" and "Qwen" in model_name else {})
    E = m.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    E = np.asarray(E, dtype=np.float32)
    np.save(f, E)
    print(f"  done in {time.time()-t0:.0f}s, dim={E.shape[1]}")
    return E


def run_eval(E, y, n_ref):
    from rsfi.whitening import SphericalWhitening
    from rsfi.geometry import RiemannianSphere
    dim = E.shape[1]
    methods = ["RAW_COS", "SVD_ONLY", "FULL(ZCA+SVD)", "LOGREG(ref)"]
    res = {m: {"auc": [], "auprc": []} for m in methods}
    for seed in range(SEEDS):
        rng = np.random.RandomState(seed)
        mal = rng.permutation(np.where(y == 1)[0])
        safe = rng.permutation(np.where(y == 0)[0])
        n_val = min(200, len(mal) - n_ref, len(safe) - n_ref)
        if n_val < 20:
            print(f"  [skip seed] мало данных после ref={n_ref}"); break
        ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
        val_idx = np.concatenate([mal[n_ref:n_ref+n_val], safe[n_ref:n_ref+n_val]])
        test_idx = np.concatenate([mal[n_ref+n_val:], safe[n_ref+n_val:]])
        y_val, y_test = y[val_idx], y[test_idx]

        sc_cos = np.array([cos_sim(E[i], RiemannianSphere.normalize(E[ref_mal].mean(axis=0))) for i in test_idx])
        res["RAW_COS"]["auc"].append(roc_auc_score(y_test, sc_cos))
        res["RAW_COS"]["auprc"].append(average_precision_score(y_test, sc_cos))

        En = RiemannianSphere.normalize(E)
        sc, _ = svd_filter_auc(En, ref_mal, ref_safe, val_idx, test_idx, y_val, y_test)
        res["SVD_ONLY"]["auc"].append(roc_auc_score(y_test, sc))
        res["SVD_ONLY"]["auprc"].append(average_precision_score(y_test, sc))

        wh = SphericalWhitening(dim=dim)
        wh.fit(En[ref_safe])
        sc, _ = svd_filter_auc(wh.transform(E), ref_mal, ref_safe, val_idx, test_idx, y_val, y_test)
        res["FULL(ZCA+SVD)"]["auc"].append(roc_auc_score(y_test, sc))
        res["FULL(ZCA+SVD)"]["auprc"].append(average_precision_score(y_test, sc))

        clf = LogisticRegression(max_iter=2000).fit(
            E[np.concatenate([ref_mal, ref_safe])],
            np.concatenate([np.ones(n_ref), np.zeros(n_ref)]))
        sc_lr = clf.decision_function(E[test_idx])
        res["LOGREG(ref)"]["auc"].append(roc_auc_score(y_test, sc_lr))
        res["LOGREG(ref)"]["auprc"].append(average_precision_score(y_test, sc_lr))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"])
    ap.add_argument("--n-ref", type=int, default=None,
                    help="размер ref (по умолчанию: 150 для ToxicChat, 50 для XSTest)")
    args = ap.parse_args()

    datasets = load_datasets()
    rows = []
    for ds_name, (texts, y) in datasets.items():
        n_ref = args.n_ref or (50 if ds_name == "xstest" else 150)
        for model_name in args.models:
            E = get_embeddings(model_name, ds_name, texts)
            print(f"\n===== {ds_name} | {model_name} (d={E.shape[1]}) | N_ref={n_ref} =====")
            res = run_eval(E, y, n_ref)
            for m, v in res.items():
                if not v["auc"]:
                    continue
                print(f"  {m:15s}: AUC {np.mean(v['auc']):.4f}±{np.std(v['auc']):.4f} | "
                      f"AUPRC {np.mean(v['auprc']):.4f}±{np.std(v['auprc']):.4f}")
                rows.append({"dataset": ds_name, "model": model_name, "dim": E.shape[1],
                             "method": m, "n_ref": n_ref,
                             "auc_mean": np.mean(v["auc"]), "auc_std": np.std(v["auc"]),
                             "auprc_mean": np.mean(v["auprc"]), "auprc_std": np.std(v["auprc"])})
    out = pd.DataFrame(rows)
    out.to_csv("e2_homogeneous_results.csv", index=False)
    print("\nSaved: e2_homogeneous_results.csv")


if __name__ == "__main__":
    main()
