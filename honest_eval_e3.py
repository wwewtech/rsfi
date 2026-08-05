"""
E3 — ОПЕРАЦИОННАЯ ТОЧКА: TPR при FPR ≤ 1% и ≤ 0.1% (порог ТОЛЬКО по валидации).

Почему это P0: для guardrail'а «первого рубежа» AUC — вторичная метрика.
Бизнесу важно: сколько атак мы ловим (TPR), если разрешено ложно банить
не более 1% (или 0.1%) легальных пользователей. Порог калибруется на
валидационной выборке и ЗАМОРАЖИВАЕТСЯ; тест видит только финальный замер.

Датасеты: wild (текущий проектный) + ToxicChat (оба лейблинга) + XSTest.
Эмбеддинги переиспользуются из кэша E2 (emb_cache/e2_*).

Метрики:
  - TPR@FPR=1% и TPR@FPR=0.1% (порог с валидации -> применён к тесту)
  - ROC-AUC, AUPRC (для справки)
  - ORACLE-строка: достижимый TPR@FPR по тестовой ROC (верхняя граница,
    помечена как oracle — для оценки цены калибровки)

Запуск:
  python honest_eval_e3.py --models all-mpnet-base-v2 BAAI/bge-base-en-v1.5
  python honest_eval_e3.py --models Qwen/Qwen3-Embedding-8B   # GPU
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from honest_eval_e2 import load_datasets, get_embeddings  # лоадеры и кэш из E2

K_LIST = [1, 5, 10, 20, 30, 40]
SEEDS = 5
TARGET_FPRS = [0.01, 0.001]


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def svd_filter_scores(E_space, ref_mal, ref_safe, val_idx, test_idx, y_val):
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
    return (np.array([-flt.evaluate(v)["rsfi"] for v in E_space[val_idx]]),
            np.array([-flt.evaluate(v)["rsfi"] for v in E_space[test_idx]]))


def operating_point(val_sc, val_y, test_sc, test_y, target_fpr):
    """Порог по валидации: квантиль safe-скоров. Замер TPR/FPR на тесте."""
    val_safe_sc = val_sc[val_y == 0]
    thr = np.quantile(val_safe_sc, 1.0 - target_fpr)
    tpr = float(np.mean(test_sc[test_y == 1] >= thr))
    fpr = float(np.mean(test_sc[test_y == 0] >= thr))
    return tpr, fpr


def oracle_tpr(test_sc, test_y, target_fpr):
    """Верхняя граница: порог по тесту (только для справки, НЕ для отчёта)."""
    thr = np.quantile(test_sc[test_y == 0], 1.0 - target_fpr)
    return float(np.mean(test_sc[test_y == 1] >= thr))


def get_scores(method, E, ref_mal, ref_safe, val_idx, test_idx, y_val):
    from rsfi.whitening import SphericalWhitening
    from rsfi.geometry import RiemannianSphere
    if method == "RAW_COS":
        c = RiemannianSphere.normalize(E[ref_mal].mean(axis=0))
        f = lambda idx: np.array([cos_sim(E[i], c) for i in idx])
        return f(val_idx), f(test_idx)
    if method == "SVD_ONLY":
        return svd_filter_scores(RiemannianSphere.normalize(E), ref_mal, ref_safe, val_idx, test_idx, y_val)
    if method == "FULL(ZCA+SVD)":
        wh = SphericalWhitening(dim=E.shape[1])
        wh.fit(RiemannianSphere.normalize(E[ref_safe]))
        return svd_filter_scores(wh.transform(E), ref_mal, ref_safe, val_idx, test_idx, y_val)
    if method == "LOGREG(ref)":
        clf = LogisticRegression(max_iter=2000).fit(
            E[np.concatenate([ref_mal, ref_safe])],
            np.concatenate([np.ones(len(ref_mal)), np.zeros(len(ref_safe))]))
        return clf.decision_function(E[val_idx]), clf.decision_function(E[test_idx])
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"])
    args = ap.parse_args()

    # датасеты: проектный wild + однородные из E2
    df = pd.read_csv("data/results/sfi_wild_10k_results.csv")
    datasets = {"wild_2000": (df["text"].tolist(),
                              (df["scenario_type"] == "MALICIOUS").astype(int).values)}
    datasets.update(load_datasets())

    rows = []
    for ds_name, (texts, y) in datasets.items():
        n_ref = 50 if ds_name == "xstest" else (150 if ds_name.startswith("toxicchat") else 200)
        for model_name in args.models:
            cache_name = ds_name if ds_name != "wild_2000" else "wild2000"
            E = get_embeddings(model_name, cache_name, texts)
            print(f"\n===== E3 | {ds_name} | {model_name} | N_ref={n_ref} =====")
            for method in ["RAW_COS", "SVD_ONLY", "FULL(ZCA+SVD)", "LOGREG(ref)"]:
                tprs = {f: [] for f in TARGET_FPRS}
                fprs = {f: [] for f in TARGET_FPRS}
                oracles = {f: [] for f in TARGET_FPRS}
                aucs, auprcs = [], []
                for seed in range(SEEDS):
                    rng = np.random.RandomState(seed)
                    mal = rng.permutation(np.where(y == 1)[0])
                    safe = rng.permutation(np.where(y == 0)[0])
                    n_val = min(200, len(mal) - n_ref, len(safe) - n_ref)
                    if n_val < 20:
                        break
                    ref_mal, ref_safe = mal[:n_ref], safe[:n_ref]
                    val_idx = np.concatenate([mal[n_ref:n_ref+n_val], safe[n_ref:n_ref+n_val]])
                    test_idx = np.concatenate([mal[n_ref+n_val:], safe[n_ref+n_val:]])
                    y_val, y_test = y[val_idx], y[test_idx]

                    val_sc, test_sc = get_scores(method, E, ref_mal, ref_safe, val_idx, test_idx, y_val)
                    aucs.append(roc_auc_score(y_test, test_sc))
                    auprcs.append(average_precision_score(y_test, test_sc))
                    for f in TARGET_FPRS:
                        tpr, fpr = operating_point(val_sc, y_val, test_sc, y_test, f)
                        tprs[f].append(tpr); fprs[f].append(fpr)
                        oracles[f].append(oracle_tpr(test_sc, y_test, f))
                if not aucs:
                    continue
                line = f"  {method:15s}: AUC {np.mean(aucs):.3f} | AUPRC {np.mean(auprcs):.3f}"
                for f in TARGET_FPRS:
                    line += (f" | TPR@FPR{f*100:g}%: {np.mean(tprs[f]):.3f}"
                             f" (реальный FPR {np.mean(fprs[f])*100:.2f}%, oracle {np.mean(oracles[f]):.3f})")
                print(line)
                for f in TARGET_FPRS:
                    rows.append({"dataset": ds_name, "model": model_name, "method": method,
                                 "target_fpr": f, "tpr_mean": np.mean(tprs[f]), "tpr_std": np.std(tprs[f]),
                                 "actual_fpr_mean": np.mean(fprs[f]), "oracle_tpr": np.mean(oracles[f]),
                                 "auc": np.mean(aucs), "auprc": np.mean(auprcs)})
    out = pd.DataFrame(rows)
    out.to_csv("e3_operating_point_results.csv", index=False)
    print("\nSaved: e3_operating_point_results.csv")


if __name__ == "__main__":
    main()
