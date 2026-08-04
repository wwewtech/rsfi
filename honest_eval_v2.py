"""
НЕЗАВИСИМАЯ ПРОВЕРКА v2 — ответ на возражение (KIMI_SOSI_TVAR.md).

Воспроизводим ТОЧНЫЙ протокол проекта (run_fitted_subspace_sweep.py):
  модель all-mpnet-base-v2 (d=768), SVD-подпространство атак (Vh[:k]),
  sweep k и N_ref. Плюс контрольные варианты:

  A) THEIR_PROTOCOL  — как у них: ZCA-whitening на первых 800 SAFE (включает
     тестовые safe -> утечка), k выбирается по ТЕСТУ (max AUC). Ожидаем ~0.87.
  B) CLEAN_PROTOCOL  — whitening ТОЛЬКО на ref-safe (без утечки),
     k выбирается по ВАЛИДАЦИИ, репорт на чистом тесте.
  C) NAIVE_COSINE    — сырой cosine к центроиду ref-атак (mpnet, 3 строки).
  D) LOGREG_REF      — LogReg, обучена только на ref (те же данные, что у RSFI).

5 сидов перестановки. Данные — реальный датасет проекта (2000 промптов).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
texts = df["text"].tolist()
y = (df["scenario_type"] == "MALICIOUS").astype(int).values
print(f"N={len(texts)}, malicious={y.sum()}, safe={len(y)-y.sum()}")

print("Encoding with all-mpnet-base-v2 (d=768)...")
t0 = time.time()
model = SentenceTransformer("all-mpnet-base-v2")
E = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
dim = E.shape[1]
print(f"encoded in {time.time()-t0:.1f}s, shape={E.shape}")

K_LIST = [1, 5, 10, 20, 30, 40]
N_REF = 200  # их максимальный N_ref (лучшие условия для метода)

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

rows = []
for seed in range(5):
    rng = np.random.RandomState(seed)
    mal_idx = rng.permutation(np.where(y == 1)[0])
    safe_idx = rng.permutation(np.where(y == 0)[0])

    ref_mal, ref_safe = mal_idx[:N_REF], safe_idx[:N_REF]
    pool_mal, pool_safe = mal_idx[N_REF:], safe_idx[N_REF:]
    # валидация (выбор k) — 200/200, чистый тест — остальное (600/600)
    val_idx = np.concatenate([pool_mal[:200], pool_safe[:200]])
    test_idx = np.concatenate([pool_mal[200:], pool_safe[200:]])
    y_val, y_test = y[val_idx], y[test_idx]

    # ---------- A) ИХ ПРОТОКОЛ: whitening на первых 800 safe (УТЕЧКА) ----------
    wh_leak = SphericalWhitening(dim=dim)
    calib800 = safe_idx[:800]  # включает вал+тест safe
    wh_leak.fit(RiemannianSphere.normalize(E[calib800]))
    Ew_leak = wh_leak.transform(E)
    S_l = RiemannianSphere.normalize(Ew_leak[ref_safe].mean(axis=0, keepdims=True))[0]
    tang_l = np.array([RiemannianSphere.log_map(S_l, v) for v in Ew_leak[ref_mal]])
    _, _, Vh_l = np.linalg.svd(tang_l, full_matrices=False)
    aucs_test = {}
    for k in K_LIST:
        flt = MultiDimensionalRSFIFilter(S_l, Vh_l[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
        sc = np.array([-flt.evaluate(v)["rsfi"] for v in Ew_leak[test_idx]])
        aucs_test[k] = roc_auc_score(y_test, sc)
    best_k_test = max(aucs_test, key=aucs_test.get)
    rows.append(("A_their_protocol(leak,k_on_test)", aucs_test[best_k_test], best_k_test))

    # ---------- B) ЧИСТЫЙ ПРОТОКОЛ: whitening только на ref-safe, k по валидации ----------
    wh = SphericalWhitening(dim=dim)
    wh.fit(RiemannianSphere.normalize(E[ref_safe]))
    Ew = wh.transform(E)
    S = RiemannianSphere.normalize(Ew[ref_safe].mean(axis=0, keepdims=True))[0]
    tang = np.array([RiemannianSphere.log_map(S, v) for v in Ew[ref_mal]])
    _, _, Vh = np.linalg.svd(tang, full_matrices=False)
    aucs_val = {}
    for k in K_LIST:
        flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
        sc = np.array([-flt.evaluate(v)["rsfi"] for v in Ew[val_idx]])
        aucs_val[k] = roc_auc_score(y_val, sc)
    best_k = max(aucs_val, key=aucs_val.get)
    flt = MultiDimensionalRSFIFilter(S, Vh[:best_k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
    sc = np.array([-flt.evaluate(v)["rsfi"] for v in Ew[test_idx]])
    rows.append(("B_clean_protocol(no_leak,k_on_val)", roc_auc_score(y_test, sc), best_k))

    # ---------- C) наивный cosine (mpnet, сырые эмбеддинги) ----------
    thr_c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
    sc = np.array([cos_sim(E[i], thr_c) for i in test_idx])
    rows.append(("C_naive_cosine_mpnet", roc_auc_score(y_test, sc), None))

    # ---------- D) LogReg на ref ----------
    clf = LogisticRegression(max_iter=2000).fit(E[np.concatenate([ref_mal, ref_safe])],
                                                np.concatenate([np.ones(N_REF), np.zeros(N_REF)]))
    rows.append(("D_logreg_ref_only", roc_auc_score(y_test, clf.decision_function(E[test_idx])), None))

print("\n===== ROC-AUC, mpnet-base-v2, N_ref=200, 5 сидов =====")
res = pd.DataFrame(rows, columns=["method", "auc", "k"])
for m, g in res.groupby("method"):
    ks = g["k"].dropna().tolist()
    print(f"  {m:38s}: {g['auc'].mean():.4f} ± {g['auc'].std():.4f}   (k: {ks})")
