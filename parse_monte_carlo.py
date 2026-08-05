import pandas as pd
import numpy as np
from pathlib import Path

def parse_and_summarize():
    input_file = Path("data/reports/monte_carlo_10h_results.csv")
    output_file = Path("data/reports/monte_carlo_summary_for_paper.csv")

    if not input_file.exists():
        print(f"[ОШИБКА] Файл {input_file} не найден.")
        return

    # 1. Загрузка сырых данных
    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} строк из {input_file.name}")

    # 2. Агрегация: среднее (mean) и стандартное отклонение (std) по 100 сидам
    # Группируем по модели, размерности (dim) и размеру выборки (n_ref)
    agg_df = df.groupby(['model', 'dim', 'n_ref']).agg({
        'auc_cos': ['mean', 'std'],
        'auc_svd': ['mean', 'std'],
        'auc_full': ['mean', 'std'],
        'auc_logreg': ['mean', 'std']
    }).reset_index()

    # Схлопываем многоуровневые заголовки столбцов
    agg_df.columns = [
        'model', 'dim', 'n_ref',
        'cos_mean', 'cos_std',
        'svd_mean', 'svd_std',
        'full_mean', 'full_std',
        'logreg_mean', 'logreg_std'
    ]

    # 3. Вывод академической таблицы в консоль
    print("\n" + "=" * 115)
    print(f"{'Модель':<40} | {'d':<5} | {'N_ref':<5} | {'Наивный COS':<12} | {'RSFI (SVD)':<12} | {'RSFI (FULL)':<12} | {'LogReg':<12}")
    print("=" * 115)

    # Сортируем модели по размерности (возрастание), затем по n_ref
    agg_df = agg_df.sort_values(by=['dim', 'model', 'n_ref'])

    for _, row in agg_df.iterrows():
        model_name = str(row['model']).split('/')[-1][:38] # Сокращаем имя для вывода
        dim = int(row['dim'])
        n_ref = int(row['n_ref'])

        # Форматирование "Mean ± Std". Если NaN, выводим "N/A"
        def fmt(m, s):
            if pd.isna(m): return "N/A".rjust(12)
            return f"{m:.4f}±{s:.3f}".rjust(12)

        cos_str = fmt(row['cos_mean'], row['cos_std'])
        svd_str = fmt(row['svd_mean'], row['svd_std'])
        full_str = fmt(row['full_mean'], row['full_std'])
        logreg_str = fmt(row['logreg_mean'], row['logreg_std'])

        # Подсветка победителя (выбор максимума из средних)
        means = {
            'COS': row['cos_mean'] if pd.notna(row['cos_mean']) else -1,
            'SVD': row['svd_mean'] if pd.notna(row['svd_mean']) else -1,
            'FULL': row['full_mean'] if pd.notna(row['full_mean']) else -1,
            'LOGREG': row['logreg_mean'] if pd.notna(row['logreg_mean']) else -1,
        }
        winner = max(means, key=means.get)
        
        # Маркируем победителя звездочкой
        if winner == 'COS': cos_str = cos_str.replace("±", "*±")
        elif winner == 'SVD': svd_str = svd_str.replace("±", "*±")
        elif winner == 'FULL': full_str = full_str.replace("±", "*±")
        elif winner == 'LOGREG': logreg_str = logreg_str.replace("±", "*±")

        print(f"{model_name:<40} | {dim:<5d} | {n_ref:<5d} | {cos_str} | {svd_str} | {full_str} | {logreg_str}")

    print("=" * 115)
    print("* - обозначает статистического лидера (наивысшее среднее AUC) в данной строке.")

    # 4. Сохранение агрегированных данных в CSV для Excel/LaTeX
    agg_df.to_csv(output_file, index=False)
    print(f"\n[УСПЕХ] Агрегированная сводная таблица сохранена в: {output_file}")

if __name__ == "__main__":
    parse_and_summarize()