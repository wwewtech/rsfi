"""
НЕЗАВИСИМАЯ ПРОВЕРКА v5 — АБЛЯЦИЯ: что даёт прирост? ZCA, SVD или их комбо?

Ответ на обвинение в манипуляции: разлагаем полный пайплайн на компоненты.
  RAW_COS      — сырой cosine к центроиду атак (без ZCA, без SVD, без сферы)
  WHITEN_ONLY  — ZCA + cosine к центроиду (есть ZCA, нет SVD)
  SVD_ONLY     — сфера + log_map + SVD-подпространство (нет ZCA)
  FULL         — ZCA + сфера + log_map + SVD (их полный метод)
k по валидации, N_ref=200/200, val 200/200, test 600/600, 5 сидов.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

MODELS = ["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]
K_LIST = [1, 5, 10, 20, 30, 40]
N_REF = 200

df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
y = (df["scenario_type"] == "MALICIOUS").astype(int).values

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def svd_filter_scores(E_space, ref_mal, ref_safe, val_idx, test_idx, y_val):
    """S = центроид ref_safe; SVD log-map ref_mal; k по val; скоры на test."""
    S = RiemannianSphere.normalize(E_space[ref_safe].mean(axis=0, keepdims=True))[0]
    tang = np.array([RiemannianSphere.log_map(S, v) for v in E_space[ref_mal]])
    _, _, Vh = np.linalg.svd(tang, full_matrices=False)
    aucs = {}
    for k in K_LIST:
        flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
        aucs[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in E_space[val_idx]])
    bk = max(aucs, key=aucs.get)
    flt = MultiDimensionalRSFIFilter(S, Vh[:bk], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
    return [-flt.evaluate(v)["rsfi"] for v in E_space[test_idx]]

for name in MODELS:
    E = np.load(f"emb_cache/{name.replace('/', '_')}.npy")
    dim = E.shape[1]
    res = {"RAW_COS": [], "WHITEN_ONLY": [], "SVD_ONLY": [], "FULL(ZCA+SVD)": []}
    for seed in range(5):
        rng = np.random.RandomState(seed)
        mal = rng.permutation(np.where(y == 1)[0])
        safe = rng.permutation(np.where(y == 0)[0])
        ref_mal, ref_safe = mal[:N_REF], safe[:N_REF]
        val_idx = np.concatenate([mal[N_REF:N_REF+200], safe[N_REF:N_REF+200]])
        test_idx = np.concatenate([mal[N_REF+200:], safe[N_REF+200:]])
        y_val, y_test = y[val_idx], y[test_idx]

        # RAW
        c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
        res["RAW_COS"].append(roc_auc_score(y_test, [cos_sim(E[i], c) for i in test_idx]))

        # WHITEN_ONLY
        wh = SphericalWhitening(dim=dim)
        wh.fit(RiemannianSphere.normalize(E[ref_safe]))
        Ew = wh.transform(E)
        cw = RiemannianSphere.normalize(Ew[ref_mal].mean(axis=0))
        res["WHITEN_ONLY"].append(roc_auc_score(y_test, [cos_sim(Ew[i], cw) for i in test_idx]))

        # SVD_ONLY (нормализация на сферу БЕЗ whitening)
        En = RiemannianSphere.normalize(E)
        res["SVD_ONLY"].append(roc_auc_score(y_test, svd_filter_scores(En, ref_mal, ref_safe, val_idx, test_idx, y_val)))

        # FULL
        res["FULL(ZCA+SVD)"].append(roc_auc_score(y_test, svd_filter_scores(Ew, ref_mal, ref_safe, val_idx, test_idx, y_val)))

    print(f"\n===== АБЛЯЦИЯ: {name} (5 сидов) =====")
    for m, v in res.items():
        print(f"  {m:16s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
