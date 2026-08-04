"""
НЕЗАВИСИМАЯ ПРОВЕРКА v3 — мультимодельный тест (честный протокол).

Для каждого эмбеддера (разная сила/архитектура/размерность):
  RSFI-SVD CLEAN  — whitening только на ref-safe, SVD-подпространство из ref-атак,
                    k выбирается по валидации (никаких утечек и подбора по тесту)
  NAIVE COSINE    — cosine к центроиду ref-атак
  LOGREG (ref)    — LogReg на тех же 400 ref-примерах (равный бюджет данных)

Протокол: N_ref=200/200, val=200/200, test=600/600, 5 сидов.
Данные: реальный датасет проекта (sfi_wild_10k_results.csv, 2000 промптов).
Эмбеддинги кэшируются в ./emb_cache/ (повторный запуск мгновенный).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

MODELS = [
    "all-MiniLM-L6-v2",                              # 384d, лёгкая (из v1)
    "paraphrase-multilingual-MiniLM-L12-v2",         # 384d, их дефолт в WildChatRunner
    "all-mpnet-base-v2",                             # 768d, их SOTA-конфиг
    "BAAI/bge-base-en-v1.5",                         # 768d, сильный retrieval-эмбеддер
]

CACHE = Path("emb_cache"); CACHE.mkdir(exist_ok=True)
df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
texts = df["text"].tolist()
y = (df["scenario_type"] == "MALICIOUS").astype(int).values

K_LIST = [1, 5, 10, 20, 30, 40]
N_REF = 200

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def get_embeddings(name):
    f = CACHE / (name.replace("/", "_") + ".npy")
    if f.exists():
        return np.load(f)
    print(f"[ENCODE] {name} ...")
    t0 = time.time()
    m = SentenceTransformer(name)
    E = m.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    np.save(f, E)
    print(f"  done in {time.time()-t0:.0f}s, dim={E.shape[1]}")
    return E

print(f"{'model':42s} {'RSFI-SVD(clean)':>16s} {'naive cos':>10s} {'logreg':>8s}")
print("-" * 82)
for name in MODELS:
    E = get_embeddings(name)
    dim = E.shape[1]
    r_rsfi, r_cos, r_lr = [], [], []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        mal = rng.permutation(np.where(y == 1)[0])
        safe = rng.permutation(np.where(y == 0)[0])
        ref_mal, ref_safe = mal[:N_REF], safe[:N_REF]
        val_idx = np.concatenate([mal[N_REF:N_REF+200], safe[N_REF:N_REF+200]])
        test_idx = np.concatenate([mal[N_REF+200:], safe[N_REF+200:]])
        y_val, y_test = y[val_idx], y[test_idx]

        wh = SphericalWhitening(dim=dim)
        wh.fit(RiemannianSphere.normalize(E[ref_safe]))
        Ew = wh.transform(E)
        S = RiemannianSphere.normalize(Ew[ref_safe].mean(axis=0, keepdims=True))[0]
        tang = np.array([RiemannianSphere.log_map(S, v) for v in Ew[ref_mal]])
        _, _, Vh = np.linalg.svd(tang, full_matrices=False)

        aucs_val = {}
        for k in K_LIST:
            flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
            aucs_val[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in Ew[val_idx]])
        bk = max(aucs_val, key=aucs_val.get)
        flt = MultiDimensionalRSFIFilter(S, Vh[:bk], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
        r_rsfi.append(roc_auc_score(y_test, [-flt.evaluate(v)["rsfi"] for v in Ew[test_idx]]))

        thr_c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
        r_cos.append(roc_auc_score(y_test, [cos_sim(E[i], thr_c) for i in test_idx]))

        clf = LogisticRegression(max_iter=2000).fit(
            E[np.concatenate([ref_mal, ref_safe])],
            np.concatenate([np.ones(N_REF), np.zeros(N_REF)]))
        r_lr.append(roc_auc_score(y_test, clf.decision_function(E[test_idx])))

    print(f"{name:42s} {np.mean(r_rsfi):>8.4f}±{np.std(r_rsfi):.4f} {np.mean(r_cos):>10.4f} {np.mean(r_lr):>8.4f}")
