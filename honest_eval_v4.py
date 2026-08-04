"""
НЕЗАВИСИМАЯ ПРОВЕРКА v4 — FEW-SHOT НИША (последний открытый вопрос).

Единственный режим, где RSFI теоретически может выиграть:
крошечный N_ref (2..50 примеров), где LogReg не может обучиться.

Сweep N_ref in [2, 5, 10, 20, 50, 200].
Методы: RSFI-SVD (k<=N_ref, k по валидации), naive cosine, LogReg(ref).
10 сидов (в few-shot режиме дисперсия высокая).
Модели: all-mpnet-base-v2, BAAI/bge-base-en-v1.5 (768d, лучшие для RSFI).
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

MODELS = ["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]
N_REFS = [2, 5, 10, 20, 50, 200]
SEEDS = 10

df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
y = (df["scenario_type"] == "MALICIOUS").astype(int).values

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

for name in MODELS:
    E = np.load(f"emb_cache/{name.replace('/', '_')}.npy")
    dim = E.shape[1]
    print(f"\n===== {name} (d={dim}), 10 сидов =====")
    print(f"{'N_ref':>6s} {'RSFI-SVD':>15s} {'naive cos':>15s} {'LogReg':>15s}")
    for n_ref in N_REFS:
        r_r, r_c, r_l = [], [], []
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            mal = rng.permutation(np.where(y == 1)[0])
            safe = rng.permutation(np.where(y == 0)[0])
            ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
            val_idx = np.concatenate([mal[200:400], safe[200:400]])
            test_idx = np.concatenate([mal[400:], safe[400:]])
            y_val, y_test = y[val_idx], y[test_idx]

            try:
                wh = SphericalWhitening(dim=dim)
                wh.fit(RiemannianSphere.normalize(E[ref_safe]))
                Ew = wh.transform(E)
                S = RiemannianSphere.normalize(Ew[ref_safe].mean(axis=0, keepdims=True))[0]
                tang = np.array([RiemannianSphere.log_map(S, v) for v in Ew[ref_mal]])
                _, _, Vh = np.linalg.svd(tang, full_matrices=False)
                k_max = min(Vh.shape[0], 40)
                k_list = sorted({k for k in [1, 5, 10, 20, 30, 40] if k <= k_max})
                aucs = {}
                for k in k_list:
                    flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                    aucs[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in Ew[val_idx]])
                bk = max(aucs, key=aucs.get)
                flt = MultiDimensionalRSFIFilter(S, Vh[:bk], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                r_r.append(roc_auc_score(y_test, [-flt.evaluate(v)["rsfi"] for v in Ew[test_idx]]))
            except Exception:
                r_r.append(np.nan)

            thr_c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
            r_c.append(roc_auc_score(y_test, [cos_sim(E[i], thr_c) for i in test_idx]))
            try:
                clf = LogisticRegression(max_iter=2000).fit(
                    E[np.concatenate([ref_mal, ref_safe])],
                    np.concatenate([np.ones(n_ref), np.zeros(n_ref)]))
                r_l.append(roc_auc_score(y_test, clf.decision_function(E[test_idx])))
            except Exception:
                r_l.append(np.nan)
        fm = lambda v: f"{np.nanmean(v):.4f}±{np.nanstd(v):.3f}"
        print(f"{n_ref:>6d} {fm(r_r):>15s} {fm(r_c):>15s} {fm(r_l):>15s}")
