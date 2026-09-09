# RSFI — что нужно сделать (TODO / next steps)

Эта заметка фиксирует текущие задачи и ограничения проекта RSFI.
Обновляется при появлении новых результатов или выявленных проблем.

## Статус (2026-09-09)

- 5 основных блоков экспериментов: E2/E3/E6/E7/E8/E9/E11 (готовы, воспроизводимы)
- E12: AdvBench + HarmBench × 4 эмбеддера × 5 сидов — готово
- E13: кросс-доменная таблица train→target (сводка + деталь) — готово
- E14: operating point и calibration — **готово** (E14_operating_point_e12.csv, 160 строк per-seed +
  E14_operating_point_summary.csv, 32 строки mean±std, генерируется `generate_e14_summary.py`)
- Тесты: **245 passed, 0 ошибок** (включая тесты E13 transfer-матрицы и E14 per-seed + summary)

## Что уже есть

### E13 — готово

- E13_cross_domain_summary.csv (сводка по mean_auc, std_auc)
- E13_cross_domain_transfer.csv (деталь: 2500 строк, 7×7 train→target × 4 эмбеддера × 5 сидов)
- Кросс-доменный эксперимент: train на {Wild/ToxicChat/XSTest/AdvBench/HarmBench}, test на всех 7 датасетах
- Сильные результаты: зеленые блоки (in-domain) почти идеальны, кросс-доменное transfer — разное по качеству

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

### 4. Aggregated results (статистика)

Сейчас есть per-seed данных (E8_delong_tests, E12 и т.д.), но нет агрегированной статистики.
Что нужно добавить:
- Summary-таблицы (в 연구 보고서 или отдельный файл)
- Агрегированные Delong-тесты
- Confidence intervals для mean_auc

## Мусор / неиспользуемое

- data/reports/monte_carlo_10h_results.csv — удалить (переместить в docs/archive)
- E5_whitening_stability.csv — пустой, удалить
- Устаревшие скрипты: honest_eval_final.py, run_*_benchmark.py — перенести в experiments/archive
- Рисунки: *_2.png — дубликаты, удалить или перенести в docs/archive

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

Технически готово: E2-E14 (полный цикл экспериментов). Осталось: секции E13/E14 в RESEARCH_REPORT.md.
