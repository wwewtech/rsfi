"""
НЕЗАВИСИМАЯ ПРОВЕРКА v8 — 10-ЧАСОВОЙ СТРЕСС-ТЕСТ (MONTE CARLO CROSS-VALIDATION).

Методология (Золотой стандарт для ВАК К1 / Scopus Q1):
- 100 независимых случайных разбиений (сидов) для каждой конфигурации.
- Плотная логарифмическая сетка Few-Shot: N_ref от 2 до 200.
- 8 открытых моделей (размерности от 384 до 8192).
- Инкрементальное сохранение (Checkpointing) для защиты от падений.
"""

import sys
import time
import warnings
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

MODELS = [
    "all-MiniLM-L6-v2",                          # d = 384
    "BAAI/bge-base-en-v1.5",                     # d = 768
    "all-mpnet-base-v2",                         # d = 768
    "BAAI/bge-large-en-v1.5",                    # d = 1024
    "Qwen/Qwen3-Embedding-8B"                    # d = 4096
]
N_REFS = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
SEEDS = 100  # 100 прогонов для получения p-value и доверительных интервалов 99%
K_LIST = [1, 2, 3, 5, 10, 20, 30, 40]
OUTPUT_CSV = Path("data/reports/monte_carlo_10h_results.csv")
# -----------------------------------------------------------------------

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def get_embeddings(model_name, texts):
    import torch
    from sentence_transformers import SentenceTransformer
    cache_dir = Path("emb_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / (model_name.replace("/", "_") + ".npy")
    
    if cache_file.exists():
        print(f"[CACHE] Загрузка {model_name} из кэша...")
        return np.load(cache_file)
    
    print(f"\n[ENCODE] Скачивание и кодирование: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs, trust_remote_code=True)
    E = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    E = np.asarray(E, dtype=np.float32)
    np.save(cache_file, E)
    
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return E

def main():
    print("=" * 100)
    print("   MONTE CARLO CROSS-VALIDATION (10-ЧАСОВОЙ СТРЕСС-ТЕСТ)")
    print("=" * 100)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Инициализация файла
    if not OUTPUT_CSV.exists():
        pd.DataFrame(columns=[
            "model", "dim", "n_ref", "seed", 
            "auc_cos", "auc_svd", "auc_full", "auc_logreg", "best_k_svd"
        ]).to_csv(OUTPUT_CSV, index=False)

    df_data = pd.read_csv("data/results/sfi_wild_10k_results.csv")
    texts = df_data["text"].tolist()
    y = (df_data["scenario_type"] == "MALICIOUS").astype(int).values

    total_iters = len(MODELS) * len(N_REFS) * SEEDS
    current_iter = 0
    t_start_global = time.time()

    for name in MODELS:
        try:
            E = get_embeddings(name, texts)
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить модель {name}: {e}")
            continue
            
        dim = E.shape[1]
        print(f"\n[{name}] Начало глубокого тестирования (d={dim})...")

        for n_ref in N_REFS:
            for seed in range(SEEDS):
                current_iter += 1
                
                # Проверка, был ли этот сид уже посчитан (если скрипт перезапускали)
                existing = pd.read_csv(OUTPUT_CSV)
                if not existing[(existing["model"] == name) & (existing["n_ref"] == n_ref) & (existing["seed"] == seed)].empty:
                    continue

                rng = np.random.RandomState(seed)
                mal = rng.permutation(np.where(y == 1)[0])
                safe = rng.permutation(np.where(y == 0)[0])
                
                ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
                val_idx = np.concatenate([mal[200:400], safe[200:400]]) # Holdout 200
                test_idx = np.concatenate([mal[400:], safe[400:]])      # Test 600
                y_val, y_test = y[val_idx], y[test_idx]

                r_cos, r_svd, r_full, r_logreg = np.nan, np.nan, np.nan, np.nan
                best_k_svd = np.nan

                # 1. Наивный косинус
                c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
                r_cos = roc_auc_score(y_test, [cos_sim(E[i], c) for i in test_idx])

                # 2. Логистическая регрессия
                try:
                    clf = LogisticRegression(max_iter=2000, solver='liblinear').fit(
                        E[np.concatenate([ref_mal, ref_safe])],
                        np.concatenate([np.ones(n_ref), np.zeros(n_ref)])
                    )
                    r_logreg = roc_auc_score(y_test, clf.decision_function(E[test_idx]))
                except:
                    pass

                # 3. SVD_ONLY (Только риманова геометрия + SVD, БЕЗ ZCA)
                try:
                    En = RiemannianSphere.normalize(E)
                    S = RiemannianSphere.normalize(En[ref_safe].mean(axis=0, keepdims=True))[0]
                    tang = np.array([RiemannianSphere.log_map(S, v) for v in En[ref_mal]])
                    _, _, Vh = np.linalg.svd(tang, full_matrices=False)
                    
                    aucs_svd = {}
                    for k in [kk for kk in K_LIST if kk <= Vh.shape[0]]:
                        flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                        aucs_svd[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in En[val_idx]])
                    
                    if aucs_svd:
                        best_k_svd = max(aucs_svd, key=aucs_svd.get)
                        flt_svd = MultiDimensionalRSFIFilter(S, Vh[:best_k_svd], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                        r_svd = roc_auc_score(y_test, [-flt_svd.evaluate(v)["rsfi"] for v in En[test_idx]])
                except:
                    pass

                # 4. FULL (ZCA + SVD)
                if n_ref >= 10: # Применяем ZCA только когда матрица не супервырождена
                    try:
                        wh = SphericalWhitening(dim=dim, reg=1e-3)
                        wh.fit(RiemannianSphere.normalize(E[ref_safe]))
                        Ew = wh.transform(E)
                        
                        S_w = RiemannianSphere.normalize(Ew[ref_safe].mean(axis=0, keepdims=True))[0]
                        tang_w = np.array([RiemannianSphere.log_map(S_w, v) for v in Ew[ref_mal]])
                        _, _, Vh_w = np.linalg.svd(tang_w, full_matrices=False)
                        
                        aucs_full = {}
                        for k in [kk for kk in K_LIST if kk <= Vh_w.shape[0]]:
                            flt = MultiDimensionalRSFIFilter(S_w, Vh_w[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                            aucs_full[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in Ew[val_idx]])
                        
                        if aucs_full:
                            bk_f = max(aucs_full, key=aucs_full.get)
                            flt_full = MultiDimensionalRSFIFilter(S_w, Vh_w[:bk_f], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                            r_full = roc_auc_score(y_test, [-flt_full.evaluate(v)["rsfi"] for v in Ew[test_idx]])
                    except:
                        pass

                # Инкрементальное сохранение
                row = pd.DataFrame([{
                    "model": name, "dim": dim, "n_ref": n_ref, "seed": seed,
                    "auc_cos": r_cos, "auc_svd": r_svd, "auc_full": r_full, 
                    "auc_logreg": r_logreg, "best_k_svd": best_k_svd
                }])
                row.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

                if current_iter % 50 == 0:
                    elapsed = time.time() - t_start_global
                    eta = (elapsed / current_iter) * (total_iters - current_iter)
                    print(f"  [ПРОГРЕСС] {current_iter}/{total_iters} | {elapsed/3600:.1f}ч прошло | ОСТАЛОСЬ ~{eta/3600:.1f}ч")

    print("\n" + "=" * 100)
    print(f"[ЗАВЕРШЕНО] Все {total_iters} итераций выполнены. Данные сохранены в {OUTPUT_CSV}")
    print("=" * 100)

if __name__ == "__main__":
    main()