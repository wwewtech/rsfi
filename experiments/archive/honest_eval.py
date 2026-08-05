"""
НЕЗАВИСИМАЯ ЧЕСТНАЯ ПРОВЕРКА RSFI (не из репозитория-проекта, написана сторонне).

Данные: data/results/sfi_wild_10k_results.csv — 2000 РЕАЛЬНЫХ промптов
(1000 джейлбрейков TrustAIRLab, 1000 безопасных WildChat).

Методология:
- stratified split: 30% калибровка (fit ZCA только на SAFE-калибровке,
  threat-векторы только из MALICIOUS-калибровки), 70% чистый тест.
- Порог НЕ подбирается по тесту. Сравнение по ROC-AUC (порогонезависимо).
- 5 сидов.

Сравниваем:
  A) Полный RSFI-пайплайн (whiten -> sphere -> log_map -> QR threat subspace) — их код.
  B) Наивный бейзлайн: cosine к центроиду угроз БЕЗ какой-либо математики.
  C) Whitening + cosine (whitening есть, "римановости" нет).
  D) LogisticRegression (обученный на калибровке) — верхняя планка.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

print("Encoding 2000 prompts (MiniLM-L6-v2)...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
E = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
print(f"encoded in {time.time()-t0:.1f}s, shape={E.shape}")

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

results = {"A_RSFI_full": [], "B_naive_cosine": [], "C_whiten_cosine": [], "D_logreg": []}
lat = {"embed_ms": [], "rsfi_ms": [], "cosine_ms": []}

for seed in range(5):
    idx_cal, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.7, stratify=y, random_state=seed
    )
    cal_safe = idx_cal[y[idx_cal] == 0]
    cal_mal = idx_cal[y[idx_cal] == 1]

    # --- общий anchor S: среднее safe-калибровки, на сфере ---
    S_raw = RiemannianSphere.normalize(E[cal_safe].mean(axis=0))

    # --- fit ZCA только на safe-калибровке ---
    wh = SphericalWhitening(dim=E.shape[1])
    wh.fit(E[cal_safe])
    Ew = np.stack([wh.transform(e) for e in E])  # whitened + on sphere

    S = RiemannianSphere.normalize(Ew[cal_safe].mean(axis=0))

    # --- A) их фильтр: threat subspace из log_map калибровочных атак ---
    V_threats = Ew[cal_mal[:20]]  # 20 атак на подпространство
    flt = MultiDimensionalRSFIFilter(S=S, V_threats=V_threats, alpha=1.0, beta=0.5)
    scores_A = np.empty(len(idx_test))
    t0 = time.perf_counter()
    for j, i in enumerate(idx_test):
        # score: чем НИЖЕ rsfi, тем опаснее -> для AUC инвертируем
        scores_A[j] = -flt.evaluate(Ew[i])["rsfi"]
    lat["rsfi_ms"].append((time.perf_counter() - t0) / len(idx_test) * 1000)
    results["A_RSFI_full"].append(roc_auc_score(y[idx_test], scores_A))

    # --- B) наивный cosine к центроиду атак, сырые эмбеддинги ---
    thr_cent_raw = RiemannianSphere.normalize(E[cal_mal].mean(axis=0))
    t0 = time.perf_counter()
    scores_B = np.array([cos_sim(E[i], thr_cent_raw) for i in idx_test])
    lat["cosine_ms"].append((time.perf_counter() - t0) / len(idx_test) * 1000)
    results["B_naive_cosine"].append(roc_auc_score(y[idx_test], scores_B))

    # --- C) whitening + cosine к центроиду атак (без log_map/QR) ---
    thr_cent_w = RiemannianSphere.normalize(Ew[cal_mal].mean(axis=0))
    scores_C = np.array([cos_sim(Ew[i], thr_cent_w) for i in idx_test])
    results["C_whiten_cosine"].append(roc_auc_score(y[idx_test], scores_C))

    # --- D) LogReg на сырых эмбеддингах (обучен на калибровке) ---
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(E[idx_cal], y[idx_cal])
    results["D_logreg"].append(roc_auc_score(y[idx_test], clf.decision_function(E[idx_test])))

    # реальная латентность эмбеддинга
    t0 = time.perf_counter()
    model.encode(texts[idx_test[0]], convert_to_numpy=True)
    lat["embed_ms"].append((time.perf_counter() - t0) * 1000)

print("\n===== ROC-AUC (mean ± std по 5 сидам), честный holdout =====")
for k, v in results.items():
    print(f"  {k:16s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

print("\n===== Реальная латентность на 1 промпт =====")
print(f"  embedding (MiniLM, CPU): {np.mean(lat['embed_ms']):.2f} ms")
print(f"  RSFI scoring:            {np.mean(lat['rsfi_ms'])*1000:.1f} µs")
print(f"  cosine scoring:          {np.mean(lat['cosine_ms'])*1000:.1f} µs")
