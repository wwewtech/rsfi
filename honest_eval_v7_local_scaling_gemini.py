"""
НЕЗАВИСИМАЯ ПРОВЕРКА v7 — ЛОКАЛЬНЫЙ МАСШТАБНЫЙ ТЕСТ (SCALING LAWS).
Тестирует локальные открытые эмбеддеры (без API) разных размерностей (от 384 до 8192)
на разных размерах обучающей выборки N_ref (от 2 до 200).

Модели (Open-Source, запускаются локально на GPU):
1. all-MiniLM-L6-v2 (d=384) - базовый контроль
2. BAAI/bge-large-en-v1.5 (d=1024) - сильный классический эмбеддер
3. Alibaba-NLP/gte-Qwen2-1.5B-instruct (d=1536) - современный LLM-эмбеддер (влезает в 12GB VRAM)
4. dunzhang/stella_en_1.5B_v5 (d=8192) - сверхвысокоразмерный эмбеддер (MRL)
"""

import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Подавление предупреждений при дефектных матрицах и несходимости LogReg
warnings.filterwarnings("ignore")

# Подключение модулей RSFI
sys.path.insert(0, "src")
from rsfi.whitening import SphericalWhitening
from rsfi.filter import MultiDimensionalRSFIFilter
from rsfi.geometry import RiemannianSphere

MODELS = [
    "all-MiniLM-L6-v2",                          # d = 384
    "BAAI/bge-large-en-v1.5",                    # d = 1024
    "Qwen/Qwen3-Embedding-8B"                    # d = 4096 (Проверено: работает стабильно)
]

N_REFS = [2, 5, 10, 50, 200]
SEEDS = 5
K_LIST = [1, 5, 10, 20, 30, 40]

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def get_embeddings(model_name, texts):
    import torch
    from sentence_transformers import SentenceTransformer
    cache_dir = Path("emb_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / (model_name.replace("/", "_") + ".npy")
    
    if cache_file.exists():
        return np.load(cache_file)
    
    print(f"\n[ENCODE] Скачивание и кодирование: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    
    # Использование float16 для экономии VRAM на RTX 3060
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs, trust_remote_code=True)
    
    E = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    E = np.asarray(E, dtype=np.float32)
    np.save(cache_file, E)
    print(f"  Готово за {time.time()-t0:.1f} сек. Размерность d={E.shape[1]}")
    
    # Очистка VRAM после использования
    del model
    torch.cuda.empty_cache()
    return E

def main():
    print("=" * 90)
    print("   ЛОКАЛЬНЫЙ ТЕСТ: ГРАНИЦЫ ПРИМЕНИМОСТИ (РАЗМЕРНОСТЬ d И РАЗМЕР ВЫБОРКИ N_ref)")
    print("=" * 90)

    # Загрузка данных
    df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
    texts = df["text"].tolist()
    y = (df["scenario_type"] == "MALICIOUS").astype(int).values

    results_table = []

    for name in MODELS:
        E = get_embeddings(name, texts)
        dim = E.shape[1]
        
        print(f"\nМодель: {name} (d={dim})")
        print(f"{'N_ref':>5s} | {'RAW_COS':>10s} | {'SVD_ONLY':>10s} | {'FULL(ZCA)':>10s} | {'LOGREG':>10s}")
        print("-" * 60)

        for n_ref in N_REFS:
            r_cos, r_svd, r_full, r_logreg = [], [], [], []
            
            for seed in range(SEEDS):
                rng = np.random.RandomState(seed)
                mal = rng.permutation(np.where(y == 1)[0])
                safe = rng.permutation(np.where(y == 0)[0])
                
                ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
                val_idx = np.concatenate([mal[200:400], safe[200:400]]) # Строгий Holdout
                test_idx = np.concatenate([mal[400:], safe[400:]])
                y_val, y_test = y[val_idx], y[test_idx]

                # 1. Наивный косинус
                c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
                r_cos.append(roc_auc_score(y_test, [cos_sim(E[i], c) for i in test_idx]))

                # 2. Логистическая регрессия (Baseline)
                try:
                    clf = LogisticRegression(max_iter=1000).fit(
                        E[np.concatenate([ref_mal, ref_safe])],
                        np.concatenate([np.ones(n_ref), np.zeros(n_ref)])
                    )
                    r_logreg.append(roc_auc_score(y_test, clf.decision_function(E[test_idx])))
                except Exception:
                    r_logreg.append(np.nan)

                # 3. SVD_ONLY (Риманова сфера + SVD, БЕЗ ZCA)
                En = RiemannianSphere.normalize(E)
                S = RiemannianSphere.normalize(En[ref_safe].mean(axis=0, keepdims=True))[0]
                tang = np.array([RiemannianSphere.log_map(S, v) for v in En[ref_mal]])
                _, _, Vh = np.linalg.svd(tang, full_matrices=False)
                
                aucs_svd = {}
                for k in [kk for kk in K_LIST if kk <= Vh.shape[0]]:
                    flt = MultiDimensionalRSFIFilter(S, Vh[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                    aucs_svd[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in En[val_idx]])
                
                bk_svd = max(aucs_svd, key=aucs_svd.get) if aucs_svd else 1
                if aucs_svd:
                    flt_svd = MultiDimensionalRSFIFilter(S, Vh[:bk_svd], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                    r_svd.append(roc_auc_score(y_test, [-flt_svd.evaluate(v)["rsfi"] for v in En[test_idx]]))
                else:
                    r_svd.append(np.nan)

                # 4. FULL (ZCA + SVD) - Опасная зона при N_ref < d
                if n_ref > 5: # При N_ref <= 5 ZCA матрица выдаст NaN/Inf без огромной регуляризации
                    try:
                        wh = SphericalWhitening(dim=dim, reg=1e-4)
                        wh.fit(RiemannianSphere.normalize(E[ref_safe]))
                        Ew = wh.transform(E)
                        
                        S_w = RiemannianSphere.normalize(Ew[ref_safe].mean(axis=0, keepdims=True))[0]
                        tang_w = np.array([RiemannianSphere.log_map(S_w, v) for v in Ew[ref_mal]])
                        _, _, Vh_w = np.linalg.svd(tang_w, full_matrices=False)
                        
                        aucs_full = {}
                        for k in [kk for kk in K_LIST if kk <= Vh_w.shape[0]]:
                            flt = MultiDimensionalRSFIFilter(S_w, Vh_w[:k], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                            aucs_full[k] = roc_auc_score(y_val, [-flt.evaluate(v)["rsfi"] for v in Ew[val_idx]])
                        
                        bk_f = max(aucs_full, key=aucs_full.get)
                        flt_full = MultiDimensionalRSFIFilter(S_w, Vh_w[:bk_f], alpha=1.5, beta=0.2, tau=0.5, is_tangent=True)
                        r_full.append(roc_auc_score(y_test, [-flt_full.evaluate(v)["rsfi"] for v in Ew[test_idx]]))
                    except Exception:
                        r_full.append(np.nan)
                else:
                    r_full.append(np.nan)

            # Форматирование вывода
            fm = lambda v: f"{np.nanmean(v):.3f}" if not np.all(np.isnan(v)) else "N/A"
            print(f"{n_ref:5d} | {fm(r_cos):>10s} | {fm(r_svd):>10s} | {fm(r_full):>10s} | {fm(r_logreg):>10s}")
            
            results_table.append({
                "model": name, "dim": dim, "n_ref": n_ref,
                "RAW_COS": np.nanmean(r_cos),
                "SVD_ONLY": np.nanmean(r_svd),
                "FULL_ZCA": np.nanmean(r_full),
                "LOGREG": np.nanmean(r_logreg)
            })

    pd.DataFrame(results_table).to_csv("scaling_laws_results.csv", index=False)
    print("\n[SUCCESS] Результаты сохранены в scaling_laws_results.csv")

if __name__ == "__main__":
    main()