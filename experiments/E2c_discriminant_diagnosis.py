"""
E2c_discriminant_diagnosis.py
================================================================================
ЗАЧЕМ ЭТОТ СКРИПТ

E2b показал: RSFI-SVD растёт монотонно с k (0.520 -> 0.875), но даже на
k=100 (половина референсного набора) не догоняет наивный косинус (0.928).
Причина, которую мы вывели математически, а не предположили: SVD в
compute_rsfi_scores() считается на ЦЕНТРИРОВАННЫХ (whitening вычитает
среднее) вредоносных эмбеддингах -> находит направления МАКСИМАЛЬНОЙ
ДИСПЕРСИИ ВНУТРИ вредоносного класса. Это не то же самое, что направление
"от безопасного к вредоносному". Ни RSFI, ни наивный косинус НИ РАЗУ не
используют безопасные примеры при построении своего направления/подпространства
-- оба by design одноклассовые.

Этот скрипт проверяет ДВЕ вещи одновременно, не смешивая их:

  (A) Whitening сам по себе, отдельно от вопроса "среднее vs SVD-подпространство".
      Раньше whitening всегда фитился на 200 вредоносных. Здесь дополнительно
      фитим его на объединении 200 вредоносных + 200 безопасных (400 всего) --
      более сбалансированная и статистически чуть более состоятельная оценка
      ковариации -- и смотрим, меняется ли что-то.

  (B) Направление/подпространство "в курсе" safe-класса, а не слепое к нему:
      - discriminant_mean: cosine к направлению mean(malicious) - mean(safe)
        вместо cosine к mean(malicious) в одиночку.
      - contrastive_svd_k: SVD не на вредоносных эмбеддингах как есть, а на
        (malicious - mean(safe)) -- те же вредоносные примеры, но смещённые
        относительно безопасного центра масс перед тем, как искать в них
        направления дисперсии. Это тот же метод, что и RSFI-SVD, но с одной
        осмысленной поправкой.

Бюджет данных одинаковый для ВСЕХ методов в этом скрипте: 200 вредоносных +
200 безопасных = 400 размеченных примеров, тот же бюджет, что у LogReg
(honest, сравнимо, а не "у LogReg data больше").

Если discriminant_mean или contrastive_svd_k хоть как-то приближаются к
LogReg (0.975 на ToxicChat), оставаясь при этом чистой линейной алгеброй без
градиентного обучения -- это первая находка за весь аудит, которая говорит
не "вот что сломано", а "вот что работает лучше, и мы можем сказать почему".
Если нет -- тоже валидный, чётный результат, и он тоже интересен: значит,
одной лишь осведомлённости о среднем safe-класса недостаточно.

ВЫВОД СКРИПТА НАМЕРЕННО ОЧЕНЬ ПОДРОБНЫЙ (per-seed, per-method, print и CSV) --
чтобы при разборе сырого вывода можно было перепроверить любую агрегированную
цифру вручную, а не верить сводке на слово.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2_homogeneous_datasets import load_toxicchat  # переиспользуем, не дублируем
from rsfi.whitening import SphericalWhitening

# ------------------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------------------
N_REF_MAL = 200
N_REF_SAFE = 200
N_SEEDS = 5
K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200]
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EPS = 1e-15


# ------------------------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------------------------
def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def fit_whitener(ref_emb: np.ndarray, dim: int) -> SphericalWhitening:
    wh = SphericalWhitening(dim=dim)
    wh.fit(ref_emb)
    return wh


# ------------------------------------------------------------------------------
# Методы направления/подпространства
# ------------------------------------------------------------------------------
def score_mean_direction(emb_eval: np.ndarray, ref_mal_emb: np.ndarray) -> np.ndarray:
    """Существующий naive_cosine: сходство со средним вредоносных. Слепой к safe."""
    mean = l2norm(ref_mal_emb.mean(axis=0, keepdims=True))
    return (l2norm(emb_eval) @ mean.T).flatten()


def score_svd_subspace(emb_eval: np.ndarray, ref_mal_emb: np.ndarray, k: int) -> np.ndarray:
    """Существующий RSFI-SVD: норма проекции на top-k дисперсию ВНУТРИ
    вредоносного класса (после L2-нормализации). Слепой к safe."""
    ref_n = l2norm(ref_mal_emb)
    emb_n = l2norm(emb_eval)
    k_eff = min(k, ref_n.shape[0], ref_n.shape[1])
    U, S, Vt = np.linalg.svd(ref_n.T, full_matrices=False)
    U_k = U[:, :k_eff]
    proj = emb_n @ U_k
    return np.linalg.norm(proj, axis=1)


def score_discriminant_mean(
    emb_eval: np.ndarray, ref_mal_emb: np.ndarray, ref_safe_emb: np.ndarray
) -> np.ndarray:
    """НОВОЕ: cosine к направлению mean(malicious) - mean(safe).
    В курсе safe-класса -- единственная разница с naive_cosine."""
    direction = ref_mal_emb.mean(axis=0) - ref_safe_emb.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + EPS)
    return l2norm(emb_eval) @ direction


def score_contrastive_svd(
    emb_eval: np.ndarray, ref_mal_emb: np.ndarray, ref_safe_emb: np.ndarray, k: int
) -> np.ndarray:
    """НОВОЕ: SVD не на вредоносных как есть, а на (malicious - mean(safe)) --
    та же дисперсия внутри класса, но измеренная относительно safe-центра,
    а не относительно центра самого вредоносного облака."""
    safe_mean = ref_safe_emb.mean(axis=0)
    displaced = ref_mal_emb - safe_mean
    k_eff = min(k, displaced.shape[0], displaced.shape[1])
    U, S, Vt = np.linalg.svd(displaced.T, full_matrices=False)
    U_k = U[:, :k_eff]
    proj = (emb_eval - safe_mean) @ U_k
    return np.linalg.norm(proj, axis=1)


# ------------------------------------------------------------------------------
# Основной цикл
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("E2c: DISCRIMINANT / WHITENING DIAGNOSIS ON TOXICCHAT")
    print("=" * 80)
    print(f"\nN_ref_malicious={N_REF_MAL}, N_ref_safe={N_REF_SAFE} "
          f"(общий бюджет 400, как у LogReg)")
    print(f"K values: {K_VALUES}")
    print(f"N_seeds: {N_SEEDS}")
    print(f"Model: {MODEL_NAME}\n")

    texts, labels = load_toxicchat()
    if len(texts) == 0:
        print("ERROR: could not load ToxicChat. Stopping.")
        return
    labels = np.array(labels)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Encoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    all_rows = []

    for seed in range(N_SEEDS):
        print(f"\n{'=' * 80}\nSEED {seed}\n{'=' * 80}")

        malicious_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]

        np.random.seed(seed)
        ref_mal_idx = np.random.choice(malicious_idx, size=N_REF_MAL, replace=False)
        ref_safe_idx = np.random.choice(safe_idx, size=N_REF_SAFE, replace=False)

        test_mal_idx = np.setdiff1d(malicious_idx, ref_mal_idx)
        test_safe_idx = np.setdiff1d(safe_idx, ref_safe_idx)
        test_idx = np.concatenate([test_mal_idx, test_safe_idx])
        y_test = labels[test_idx]

        print(f"  ref_mal={len(ref_mal_idx)}  ref_safe={len(ref_safe_idx)}  "
              f"test={len(test_idx)} ({y_test.sum()} malicious, {(1 - y_test).sum()} safe)")

        emb_test = embeddings[test_idx]
        ref_mal_raw = embeddings[ref_mal_idx]
        ref_safe_raw = embeddings[ref_safe_idx]

        # Whitener, обученный на ОБЪЕДИНЕНИИ 400 референсных примеров
        # (не только на 200 вредоносных, как в оригинальном compute_rsfi_scores)
        ref_combined_raw = np.vstack([ref_mal_raw, ref_safe_raw])
        wh_combined = fit_whitener(ref_combined_raw, dim=dim)

        emb_test_wh = wh_combined.transform(emb_test)
        ref_mal_wh = wh_combined.transform(ref_mal_raw)
        ref_safe_wh = wh_combined.transform(ref_safe_raw)

        def log_and_store(method_name, scores, extra=None):
            auc = roc_auc_score(y_test, scores)
            all_rows.append({
                "seed": seed, "method": method_name, "roc_auc": auc,
                "n_test": len(test_idx), **(extra or {}),
            })
            print(f"    {method_name:38s} AUC={auc:.4f}")
            return auc

        print("\n  -- Группа A: существующие методы (слепые к safe) --")
        log_and_store("A1_naive_cosine_raw", score_mean_direction(emb_test, ref_mal_raw))
        log_and_store("A1b_cosine_whitened400", score_mean_direction(emb_test_wh, ref_mal_wh))
        for k in K_VALUES:
            log_and_store(f"A2_rsfi_svd_raw_k{k}",
                          score_svd_subspace(emb_test, ref_mal_raw, k), {"k": k})
        for k in K_VALUES:
            log_and_store(f"A3_rsfi_svd_whitened400_k{k}",
                          score_svd_subspace(emb_test_wh, ref_mal_wh, k), {"k": k})

        print("\n  -- Группа B: новые safe-aware методы --")
        log_and_store("B1_discriminant_mean_raw",
                      score_discriminant_mean(emb_test, ref_mal_raw, ref_safe_raw))
        log_and_store("B1b_discriminant_mean_whitened",
                      score_discriminant_mean(emb_test_wh, ref_mal_wh, ref_safe_wh))
        for k in K_VALUES:
            log_and_store(f"B2_contrastive_svd_raw_k{k}",
                          score_contrastive_svd(emb_test, ref_mal_raw, ref_safe_raw, k), {"k": k})
        for k in K_VALUES:
            log_and_store(f"B2b_contrastive_svd_whitened_k{k}",
                          score_contrastive_svd(emb_test_wh, ref_mal_wh, ref_safe_wh, k), {"k": k})

        print("\n  -- Группа C: супервизионный потолок (тот же бюджет 400) --")
        X_train = np.vstack([ref_mal_raw, ref_safe_raw])
        y_train = np.concatenate([np.ones(len(ref_mal_idx)), np.zeros(len(ref_safe_idx))])
        logreg = LogisticRegression(max_iter=1000, random_state=seed)
        logreg.fit(X_train, y_train)
        log_and_store("C1_logreg_raw", logreg.decision_function(emb_test))

        logreg_wh = LogisticRegression(max_iter=1000, random_state=seed)
        logreg_wh.fit(np.vstack([ref_mal_wh, ref_safe_wh]), y_train)
        log_and_store("C1b_logreg_whitened", logreg_wh.decision_function(emb_test_wh))

    # --------------------------------------------------------------------------
    # Сохранение и сводка
    # --------------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    out_path = Path(__file__).parent.parent / "data" / "results" / "E2c_discriminant_diagnosis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n\nRaw per-seed results saved to: {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: mean +- std по 5 сидам, отсортировано по AUC")
    print("=" * 80)
    summary = (
        df.groupby("method")["roc_auc"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary.round(4).to_string())

    print("\n" + "=" * 80)
    print("КЛЮЧЕВЫЕ СРАВНЕНИЯ ДЛЯ ИНТЕРПРЕТАЦИИ")
    print("=" * 80)
    print("1) A1_naive_cosine_raw  vs  B1_discriminant_mean_raw")
    print("   -> помогает ли ОДНО ЛИШЬ знание mean(safe), без всякого SVD?")
    print("2) A2_rsfi_svd_raw_k20  vs  B2_contrastive_svd_raw_k20")
    print("   -> помогает ли safe-центрирование именно SVD-подпространству на k=20?")
    print("3) A2_rsfi_svd_raw_k*  vs  A3_rsfi_svd_whitened400_k*")
    print("   -> меняет ли что-то whitening, обученный на 400 (а не 200) референсных?")
    print("4) B1/B2 (без градиентного обучения)  vs  C1_logreg_raw (с обучением)")
    print("   -> насколько близко чистая линейная алгебра подходит к обученному классификатору")
    print("      на ОДИНАКОВОМ бюджете данных (400 размеченных примеров)?")


if __name__ == "__main__":
    main()
