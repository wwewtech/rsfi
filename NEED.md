# RSFI — что нужно сделать (TODO / next steps)

Эта заметка фиксирует текущие задачи и ограничения проекта RSFI.
Обновляется при появлении новых результатов или выявленных проблем.

## Статус (2026-09-08)

- 5 основных блоков экспериментов: E2/E3/E6/E7/E8/E9/E11 (готовы, воспроизводимы)
- E12: AdvBench + HarmBench × 4 эмбеддера × 5 сидов — готово
- E13: кросс-доменная таблица train→target (сводка + деталь) — готово
- E14: operating point и calibration — **данные отсутствуют**, скрипт существует
- Тесты: 243 passed, 2 ошибки (только E14 — файла нет)

## Что уже есть

### E13 — готово

- E13_cross_domain_summary.csv (сводка по mean_auc, std_auc)
- E13_cross_domain_transfer.csv (деталь: 2500 строк, 7×7 train→target × 4 эмбеддера × 5 сидов)
- Кросс-доменный эксперимент: train на {Wild/ToxicChat/XSTest/AdvBench/HarmBench}, test на всех 7 датасетах
- Сильные результаты: зеленые блоки (in-domain) почти идеальны, кросс-доменное transfer — разное по качеству

## Текущие задачи

### 1. E14 — получить данные (критично)

Скрипт `experiments/E14_operating_point_e12.py` существует, но данные не сгенерированы.
Причина: скрипт долго выполняется (более 30 секунд на CPU), при этом все кэши эмбеддеров уже загружены.

Что сделано:
- [ ] Запустить E14 на GPU (если CUDA доступно: `torch.cuda.is_available()` → True)
- [ ] Если CPU — уменьшить количество seed для Qwen (4096d) до 1-2
- [ ] Проверить результаты

### 2. Тесты консистентности для E13 (следующий шаг)

- Сейчас тесты E12/E13/E14 уже в test_report_consistency.py (204 строки добавлено)
- Но только E12 проходит, E13 и E14 — ошибки (нет данных)
- После E14 нужно:
  - Добавить тесты для E13_summary и E13_transfer (проверка shape, schema, нулевых значений)
  - Добавить тесты для E14_operating_point (shape, schema, bounds)

### 3. REPORT.md — обновить результатами E13

Сейчас REPORT.md не содержит E13 результатов.
После E14 нужно:
- Добавить секцию "E13: Cross-Domain Transfer" в RESEARCH_REPORT.md
- Добавить секцию "E14: Operating Point" (когда данные готовы)

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
- Не забыть про E12 — уже сделано, но в commit ещё не попало
- E14 — главная брешь в данных сейчас

Технически готово: E2-E13. E14 — последний недостающий кусок.
