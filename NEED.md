# RSFI — что нужно сделать (TODO / next steps)

Эта заметка фиксирует текущие задачи и ограничения проекта RSFI.
Обновляется при появлении новых результатов или выявленных проблем.

## Статус (2026-09-16)

- 5 основных блоков экспериментов: E2/E3/E6/E7/E8/E9/E11 (готовы, воспроизводимы)
- E12: AdvBench + HarmBench × 4 эмбеддера × 5 сидов — готово
- E13: кросс-доменная таблица train→target (сводка + деталь) — готово; аудит утечки диагонали исправлен (2026-09-09)
- E14: operating point и calibration — **готово** (E14_operating_point_e12.csv, 160 строк per-seed +
  E14_operating_point_summary.csv, 32 строки mean±std, генерируется `generate_e14_summary.py`)
- Тесты: **после добавления E6b-translit и агрегатов — 257 passed, 0 ошибок** (см. сводку в §8 отчёта)
- **E6b-translit (2026-09-16)**: смена раскладки ЙЦУКЕН↔QWERTY добавлена в `OBFUSCATIONS` как 7-й класс
  обфускации и прогнана по всем 3 датасетам × 3 эмбеддерам × 6 методам. Результат: самый разрушительный
  из шести классов — на mpnet+XSTest полярность скора инвертируется (A1 0.0001, A2 0.0006); B1 —
  наименее хрупкий (средний провал −0.40 против −0.55 у A2). Актуально для §6.1 (Таблица 8, факт 7).
- **Агрегированная статистика (need.md #4, 2026-09-16)**: `experiments/generate_aggregated_summary.py`
  пересчитывает из per-seed CSV 95% CI mean AUC и агрегированные DeLong-тесты;
  выводы `data/results/AGGREGATED_mean_auc_ci.csv` (674 группы) и `AGGREGATED_delong.csv` (196 агрегатов);

## Что уже есть

### E13 — готово (аудит утечки диагонали исправлен 2026-09-09)

- E13_cross_domain_summary.csv (сводка по mean_auc, std_auc)
- E13_cross_domain_transfer.csv (деталь: 2500 строк, 5×5 train→target × 4 эмбеддера × 5 сидов; XSTest-диагональ пропущена — вырождена: все 200 позитивов помещаются в ref-пул)
- Кросс-доменный эксперимент: train на {Wild/ToxicChat/XSTest/AdvBench/HarmBench}, test на всех 5 датасетах
- Аудит 2026-09-09: на диагонали (train == target) `balanced_eval_idx` тем же сидом выбирала те же ячейки, что и ref-пул — 100% перекрытие, in-domain AUC был завышен до ≈0.999. Исправлено: ref-пул исключается из теста (`setdiff`, семантики E12/E14). Новые in-domain средние: A1 0.889 / B1 **0.957** / B1b 0.949 / B1w 0.956. Кросс-домен не изменился (0.754/0.783/0.748/0.761) — выводы §6.5 отчёта стоят; теперь B1 — лучший метод в обоих режимах.

## Текущие задачи

### 1. E14 — ✅ ВЫПОЛНЕНО (2026-09-09)

- [x] Запустить E14 (4 эмбеддера × 5 сидов; детерминированно из кэшей emb_cache, GPU для пересчёта не нужен)
- [x] Сгенерировать per-seed CSV (160 строк: 2 датасета × 4 эмбеддера × seeds × 4 метода)
- [x] Сгенерировать summary-CSV (32 строки mean±std) через `experiments/generate_e14_summary.py`
- [x] Тесты E14 в test_report_consistency.py (schema, monotonicity, bounds, summary-consistency)
- Примечание: Qwen3-Embedding-8B (4096d) — 5 сидов в данных; первичное кодирование требует GPU/offload (~16 ГБ bf16 на 12-ГБ карте), повторный запуск идёт из кэшей. CPU-fallback: `N_SEEDS_QWEN=1`

### 2. Тесты консистентности E13/E14 — ✅ ВЫПОЛНЕНО (2026-09-09)

- [x] Тесты E13 transfer-матрицы (полнота train×target×embedder, сида, сани средних)
- [x] Тесты E14 per-seed (schema, bounds, монотонность TPR)
- [x] Тесты E14 summary (32×11 shape, пересчёт из per-seed, границы значений)

### 3. REPORT.md — обновить результатами E13/E14 (оставшееся)

Сейчас REPORT.md не содержит E13/E14 результатов. Нужно:
- Добавить секцию "E13: Cross-Domain Transfer" в RESEARCH_REPORT.md
- Добавить секцию "E14: Operating Point & Calibration" (данные готовы)

### 4. Aggregated results (статистика) — ✅ ВЫПОЛНЕНО (2026-09-16)

- [x] Скрипт `experiments/generate_aggregated_summary.py` (детерминированный пересчёт из per-seed CSV)
- [x] Summary-таблицы с 95% CI mean AUC: `data/results/AGGREGATED_mean_auc_ci.csv` (674 агрегата:
      E2d/E2q, E8/E8q, E12, E13 transfer, E13 cross-domain, E14)
- [x] Агрегированные DeLong-тесты: `data/results/AGGREGATED_delong.csv` (196 групп
      источник × датасет × модель × пара: wins/ties/losses, средний и максимальный p, доля p<0.05)
- [x] Секции 4.1 (95% CI) и 4.2 (агрегированные DeLong) в RESEARCH_REPORT.md
- [x] Тесты `tests/test_aggregated_statistics.py` (пересчёт выборки ячеек из per-seed CSV)
- Формула CI: $\bar{x} \pm t_{0.975,n-1}\cdot s/\sqrt{n}$ — честно отмечаем, что у насыщенных AUC
  (~0.9999) верхняя граница может выходить за 1.0 (в отчёте: §4.1, HarmBench/Qwen B1w).

## Мусор / неиспользуемое — ✅ всё обработано (2026-09-16)

- [x] `data/reports/monte_carlo_10h_results.csv` — удалён ранее (перемещён в docs/archive)
- [x] `E5_whitening_stability.csv` — пустой, удалён
- [x] Устаревшие скрипты: `honest_eval_final.py`, `run_*_benchmark.py` — перенесены в experiments/archive
- [x] Рисунки: `*_2.png` — дубликаты, перенесены в docs/archive/figures_legacy

## Будущие эксперименты (опционально)

### 1. Defense-aware attacks (E6c-style)

- E6c: defense-aware adaptive attack — уже есть основные результаты
- Potentailly: extend to E12 data (AdvBench, HarmBench)

### 2. Cross-domain benchmark expansion

- Добавить еще датасеты (например, BigBench).

### 3. LLM-as-a-Judge (Bendera для E2e / E11)

- Оригинальный plan: E2e extension with LLM judges
- E12 — первый шаг (AdvBench, HarmBench)


## Примечания

- Все результаты воспроизводимы из кэшей (emb_cache), без повторного embedding
- E14 использует только кэши эмбеддеров (emb_cache) — детерминирован, GPU не нужен
- Файлы в data/results игнорируются git'ом — при коммите нужен `git add -f`
- E13 и E14 закоммичены и запушены (коммиты ce78c86, 1a000f3)

Технически готово: E2-E14 + E6b-translit + агрегированная статистика (полный цикл экспериментов).
