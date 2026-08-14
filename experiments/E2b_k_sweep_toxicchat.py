"""
E2b_k_sweep_toxicchat.py
=========================
Проверка гипотезы из аудита: на ToxicChat разрыв RSFI vs cosine (0.769 vs
0.928) может объясняться не "коллапсом на гомогенных данных", а тем, что
k=20-мерное SVD-подпространство при shrinkage~0.22 размывает сигнал,
который одно направление (ближе к тому, что делает cosine) ловит чище.

Использует ТОТ ЖЕ протокол, что и E2_homogeneous_datasets.py (n_ref=200,
n_val=200, 5 сидов, тот же split), и импортирует load_toxicchat() /
compute_rsfi_scores() / compute_naive_cosine_scores() оттуда напрямую —
чтобы не было риска, что скопированная логика случайно разойдётся
с оригиналом и исказит сравнение. Положить в тот же experiments/, что и
E2_homogeneous_datasets.py.

Если гипотеза верна: AUC должен расти при уменьшении k, и k=1 должен
быть ближе к cosine, чем k=20.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from E2_homogeneous_datasets import (
    load_toxicchat,
    compute_rsfi_scores,
    compute_naive_cosine_scores,
)

K_VALUES = [1, 2, 5, 10, 20, 50, 100]
N_REF = 200
N_VAL = 200
N_SEEDS = 5
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def main():
    print("=" * 80)
    print("E2b: K-SWEEP ON TOXICCHAT")
    print("=" * 80)
    print(f"\nK values: {K_VALUES}")
    print(f"N_ref={N_REF}, N_val={N_VAL}, N_seeds={N_SEEDS}, model={MODEL_NAME}")

    texts, labels = load_toxicchat()
    if len(texts) == 0:
        print("\nERROR: could not load ToxicChat. Stopping.")
        return
    labels = np.array(labels)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    results = []
    for seed in range(N_SEEDS):
        malicious_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]

        np.random.seed(seed)
        ref_idx = np.random.choice(malicious_idx, size=N_REF, replace=False)
        remaining_mal_idx = np.setdiff1d(malicious_idx, ref_idx)
        val_mal_idx = np.random.choice(remaining_mal_idx, size=N_VAL // 2, replace=False)
        val_safe_idx = np.random.choice(safe_idx, size=N_VAL // 2, replace=False)
        val_idx = np.concatenate([val_mal_idx, val_safe_idx])
        test_idx = np.setdiff1d(np.arange(len(texts)), np.concatenate([ref_idx, val_idx]))
        y_test = labels[test_idx]

        combined = np.vstack([embeddings[ref_idx], embeddings[test_idx]])

        # cosine не зависит от k — считаем один раз на сид, как опорную линию
        scores_cos = compute_naive_cosine_scores(combined, np.arange(N_REF))[N_REF:]
        auc_cos = roc_auc_score(y_test, scores_cos)

        row = {"seed": seed, "n_test": len(test_idx), "cosine_auc": auc_cos}

        line = f"  seed={seed}: cosine={auc_cos:.4f}"
        for k in K_VALUES:
            scores_rsfi = compute_rsfi_scores(
                combined, np.arange(N_REF), k=k, apply_whitening=True
            )[N_REF:]
            auc_rsfi = roc_auc_score(y_test, scores_rsfi)
            row[f"rsfi_k{k}_auc"] = auc_rsfi
            line += f"  k{k}={auc_rsfi:.4f}"

        results.append(row)
        print(line)

    df = pd.DataFrame(results)
    out_path = Path(__file__).parent.parent / "data" / "results" / "E2b_k_sweep_toxicchat.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (mean +- std по 5 сидам)")
    print("=" * 80)
    print(f"  cosine (опора, не зависит от k): {df['cosine_auc'].mean():.4f} +- {df['cosine_auc'].std():.4f}")
    for k in K_VALUES:
        col = f"rsfi_k{k}_auc"
        print(f"  RSFI k={k:3d}:                      {df[col].mean():.4f} +- {df[col].std():.4f}")

    print(f"\nSaved to: {out_path}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("Если AUC растёт при уменьшении k и k=1 близко к cosine -> гипотеза")
    print("о размытии сигнала избыточной размерностью подпространства подтверждается.")
    print("Если AUC не зависит от k заметно -> дело не в k, ищем другое объяснение")
    print("(например, в самом shrinkage или в структуре whitening).")


if __name__ == "__main__":
    main()
