"""
НЕЗАВИСИМАЯ ПРОВЕРКА v6 — ВЫСОКОРАЗМЕРНЫЕ ЭМБЕДДИНГИ (3072d / 4096d).

Запускать на машине с GPU (8B-модель в bf16 ~16-18 GB VRAM).

Рекомендуемые модели (меняются флагом --models):
  Qwen/Qwen3-Embedding-8B              — 4096d, специализированный эмбеддер
  Alibaba-NLP/gte-Qwen2-7B-instruct    — 3584d
  Salesforce/SFR-Embedding-Mistral     — 4096d
(ровно 3072d — это OpenAI text-embedding-3-large, только через API;
 при желании добавьте свой эмбеддер любой размерности в --models)

ВАЖНО (математика): при d=4096 и N_ref=200 ковариационная матрица
вырождена (ранг <= 199 << 4096), поэтому ZCA-whitening в этом режиме
статистически бессмыслен. Скрипт честно показывает это: сравнивает
FULL (ZCA+SVD) и SVD_ONLY (без ZCA) — смотрите разницу сами.

Протокол (честный, как в v2-v5): whitening только на ref-safe,
k по валидации, тест не используется ни для чего, кроме финального AUC.
N_ref=200/200, val=200/200, test=600/600, 5 сидов.

Запуск:
  pip install -U sentence-transformers torch
  python honest_eval_v6.py --models Qwen/Qwen3-Embedding-8B
Результат: высокоразмерная таблица AUC + high_dim_results.csv
"""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

K_LIST = [1, 5, 10, 20, 30, 40]
N_REF = 200
SEEDS = 5


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
    return roc_auc_score(y_test, [-flt.evaluate(v)["rsfi"] for v in E_space[test_idx]]), bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-Embedding-8B"])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
    texts = df["text"].tolist()
    y = (df["scenario_type"] == "MALICIOUS").astype(int).values

    cache = Path("emb_cache"); cache.mkdir(exist_ok=True)
    rows = []

    for name in args.models:
        f = cache / (name.replace("/", "_") + ".npy")
        if f.exists():
            E = np.load(f)
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[ENCODE] {name} on {dev} ...")
            t0 = time.time()
            m = SentenceTransformer(name, device=dev,
                                    model_kwargs={"torch_dtype": "bfloat16"} if dev == "cuda" else {})
            E = m.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=False)
            E = np.asarray(E, dtype=np.float32)
            np.save(f, E)
            print(f"  done in {time.time()-t0:.0f}s, dim={E.shape[1]}")
        dim = E.shape[1]
        print(f"\n===== {name} (d={dim}) =====")

        from rsfi.whitening import SphericalWhitening
        from rsfi.geometry import RiemannianSphere

        res = {"RAW_COS": [], "SVD_ONLY": [], "FULL(ZCA+SVD)": [], "LOGREG(ref)": []}
        ks = []
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            mal = rng.permutation(np.where(y == 1)[0])
            safe = rng.permutation(np.where(y == 0)[0])
            ref_mal, ref_safe = mal[:N_REF], safe[:N_REF]
            val_idx = np.concatenate([mal[N_REF:N_REF+200], safe[N_REF:N_REF+200]])
            test_idx = np.concatenate([mal[N_REF+200:], safe[N_REF+200:]])
            y_val, y_test = y[val_idx], y[test_idx]

            c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
            res["RAW_COS"].append(roc_auc_score(y_test, [cos_sim(E[i], c) for i in test_idx]))

            En = RiemannianSphere.normalize(E)
            auc_svd, bk = svd_filter_auc(En, ref_mal, ref_safe, val_idx, test_idx, y_val, y_test)
            res["SVD_ONLY"].append(auc_svd); ks.append(bk)

            # FULL: внимание — при d >> N_ref ковариация вырождена (ранг <= N_ref-1)
            t0 = time.time()
            wh = SphericalWhitening(dim=dim)
            wh.fit(RiemannianSphere.normalize(E[ref_safe]))
            Ew = wh.transform(E)
            auc_full, _ = svd_filter_auc(Ew, ref_mal, ref_safe, val_idx, test_idx, y_val, y_test)
            res["FULL(ZCA+SVD)"].append(auc_full)
            print(f"  seed {seed}: SVD_ONLY={auc_svd:.4f} FULL={auc_full:.4f} (zca fit {time.time()-t0:.0f}s)")

            clf = LogisticRegression(max_iter=2000).fit(
                E[np.concatenate([ref_mal, ref_safe])],
                np.concatenate([np.ones(N_REF), np.zeros(N_REF)]))
            res["LOGREG(ref)"].append(roc_auc_score(y_test, clf.decision_function(E[test_idx])))

        for meth, v in res.items():
            print(f"  {meth:15s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        print(f"  (k по валидации: {ks})")
        for meth, v in res.items():
            rows.append({"model": name, "dim": dim, "method": meth,
                         "auc_mean": np.mean(v), "auc_std": np.std(v)})

    out = pd.DataFrame(rows)
    out.to_csv("high_dim_results.csv", index=False)
    print("\nSaved: high_dim_results.csv")


if __name__ == "__main__":
    main()
