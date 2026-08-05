# Реалистичный план дальнейших действий для проекта RSFI

**Цель:** Довести исследование до честного состояния, пригодного для защиты диплома и подачи в научный журнал (уровень К3 / workshop реалистичен, К1 маловероятен).

**Временные рамки:** 2-4 недели активной работы

---

## Приоритет 0: Критические эксперименты (без них — отклонение)

### E2: Однородные датасеты (оценка: 3-5 дней)

**Проблема:** Текущий датасет стилистически неоднороден — длинные DAN-шаблоны легко отделить от обычных промптов по длине текста. Любой метод (даже `len(text) > 100`) покажет завышенный AUC.

**Что делать:**
```python
# 1. ToxicChat-0124 от LMSYS
from datasets import load_dataset
ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
# Фильтр: toxicity==1 OR jailbreaking==1 → malicious
# toxicity==0 AND jailbreaking==0 → safe
# Ожидаемый размер: ~2000-2500 примеров

# 2. XSTest-v2 (контрастные пары)
# Скачать с https://github.com/paul-rottger/exaggerated-safety
# 250 safe ("kill the process") + 200 unsafe ("kill the person")
# Это УБИЙЦА геометрических фильтров: cosine к "kill" даст высокий FPR

# 3. Прогон честного протокола
for dataset in [toxicchat, xstest, current_wild]:
    results = run_honest_protocol(
        dataset=dataset,
        models=["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"],
        n_ref=200,
        n_val=200,
        seeds=5,
        methods=["RSFI-SVD", "naive_cosine", "LogReg"]
    )
    # Репорт: ROC-AUC, PR-AUC, таблица по датасетам
```

**Ожидаемый результат:**
- На ToxicChat/XSTest все методы просядут (геометрические до 0.6-0.7)
- **Главный вопрос:** просядет ли RSFI МЕНЬШЕ косинуса? Если да — это и есть вклад.

**Файл:** `experiments/E2_homogeneous_datasets.py`  
**Вывод:** `results/E2_homogeneous_results.csv` + таблица в статью

---

### E3: Операционная точка (оценка: 1-2 дня)

**Проблема:** Для guardrail важна метрика **TPR при FPR ≤ 1%** (сколько атак поймали, не превысив 1% ложных банов), а не общий AUC.

**Что делать:**
```python
# Калибровка порога ТОЛЬКО по валидации
val_scores = compute_scores(val_set)
threshold = find_threshold_for_fpr(val_scores, target_fpr=0.01)

# Оценка на тесте с этим порогом
test_scores = compute_scores(test_set)
tpr_at_1pct = compute_tpr(test_scores, threshold)
tpr_at_01pct = compute_tpr(test_scores, find_threshold_for_fpr(val_scores, 0.001))

# PR-AUC (precision-recall curve, стандарт GradSafe)
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_test, scores)
```

**Таблица для статьи:**
| Метод | ROC-AUC | PR-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|-------|---------|--------|------------|--------------|
| RSFI-SVD | 0.821 | ? | ? | ? |
| Naive cosine | 0.785 | ? | ? | ? |
| LogReg | 0.874 | ? | ? | ? |

**Файл:** `experiments/E3_operating_point.py`

---

### E7: Внешние бейзлайны на общих данных (оценка: 5-7 дней)

**Проблема:** Сравнение "мы 0.88, Llama Guard 0.92" бессмысленно, если данные разные. Нужен head-to-head на ОДНИХ данных.

**Что делать:**
```python
# 1. Установить бейзлайны
pip install transformers protectai-guardrails

# 2. Загрузить модели
from transformers import AutoModelForSequenceClassification, AutoTokenizer

models = {
    "Meta Prompt-Guard-86M": "meta-llama/Prompt-Guard-86M",
    "ProtectAI deberta": "protectai/deberta-v3-base-prompt-injection",
    # ITMO codebook реализовать вручную (k-NN cosine, paper: arXiv:2604.25716)
}

# 3. Прогон на ОДНОМ датасете (sfi_wild_10k_results.csv)
# ОДИНАКОВЫЙ split (те же индексы ref/val/test)
# Репорт: таблица ROC-AUC, PR-AUC, latency

# 4. Честно признать, где RSFI проигрывает
```

**Файл:** `experiments/E7_external_baselines.py`  
**Вывод:** `results/E7_head_to_head.csv`

---

## Приоритет 1: Усиливающие эксперименты

### E5: Стабильность whitening в high-dim (2-3 дня)

**Гипотеза:** При d=4096 и N_calib=200 ковариация вырождена (ранг ≤ 199). Вклад ZCA должен вырасти при N_calib > d.

```python
# Sweep N_calib на Qwen3-Embedding-8B (4096d)
for n_calib in [200, 500, 1000, 5000, 10000]:
    wh = SphericalWhitening(dim=4096)
    wh.fit(safe_corpus[:n_calib])
    # Измерить: condition number, вклад ZCA к AUC
```

**Файл:** `experiments/E5_whitening_stability.py`

---

### E6: Адаптивные атаки (3-5 дней)

**Граница применимости:** Обфусцированные атаки (base64, leetspeak) частично коллапсируют на многообразие clean промптов → геометрические фильтры уязвимы.

```python
# 1. GCG-суффиксы (adversarial optimization)
# pip install nanogcg
# 2. Obfuscation: base64, rot13, leetspeak, zero-width chars
# 3. Multilingual: yanismiraoui/MultiJail

# Ожидание: AUC падёт до 0.6-0.7
# ЭТО НЕ ПРОВАЛ — это честное измерение границы
```

**Файл:** `experiments/E6_adaptive_attacks.py`

---

### E10: Статистическая значимость (1-2 дня)

```python
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests

# 1. DeLong test для разницы AUC
pval_rsfi_vs_cosine = delong_test(y_test, scores_rsfi, scores_cosine)
pval_rsfi_vs_logreg = delong_test(y_test, scores_rsfi, scores_logreg)

# 2. Bootstrap 95% CI
ci = bootstrap((scores_rsfi,), statistic=roc_auc_score, n_resamples=1000)

# 3. 10 сидов вместо 5
# 4. Поправка Холма на множественные сравнения
pvals_corrected = multipletests(pvals, method='holm')
```

**Файл:** `experiments/E10_statistical_tests.py`

---

## Исправления документации (1-2 дня)

### README.md / README_RU.md

**Убрать:**
```diff
- | **Разделимость (ROC-AUC)**| 0.92 | 0.96 | 0.89 | **1.0000** |
+ | **Разделимость (ROC-AUC)**| 0.92 | 0.96 | 0.89 | **0.856** (4096d) |

- * ⚡ **Сверхбыстрая валидация (< 10 мс):** Расчет индекса для одного вектора занимает всего **21 микросекунду**.
+ * ⚡ **Быстрая валидация:** Полная стоимость пайплайна (эмбеддинг + фильтрация) ~10-15 мс на CPU.

- * 🛡️ **Zero-Shot & Black-Box API Protection:** Не требует доступа к весам LLM. Не нуждается в дообучении сторонних классификаторов-дискриминаторов.
+ * 🛡️ **Few-Shot & Black-Box:** Не требует доступа к весам LLM. Не нуждается в дообучении нейросети (требует 50-200 размеченных примеров для калибровки).
```

### docs/math.md

**Исправить противоречие порогов:**
```diff
- $$\tau \in \mathbb{R}$$ - калиброванный порог принятия решений (на реальных данных $\tau^* = +0.65$).
+ $$\tau \in \mathbb{R}$$ - порог принятия решений, калибруется по валидационной выборке для целевого FPR.

- ($\tau^* = 0.0$).
+ (порог зависит от целевого FPR, типично $\tau \in [-0.5, 0.5]$).

- Рост $k=1 \dots 40$ увеличивает покрытие атак до 93.7% и ROC-AUC до **0.8783** при задержке **22.1 мкс**.
+ Рост $k=1 \dots 40$ увеличивает покрытие атак до 93.7% и ROC-AUC до **0.856** (честная оценка на d=4096) при стоимости фильтрации **~20 мкс** (полный пайплайн с эмбеддингом ~10-15 мс).
```

### docs/benchmarks.md

**Удалить / переместить в legacy:**
```diff
- ## HarmBench CAIS 2024
- **Accuracy: 100.00%**
- **False Positive Rate: 0.00%**
+ (УДАЛИТЬ — датасет фейковый, 9 уникальных текстов)
```

### Добавить новый раздел: `docs/limitations.md`

```markdown
# Границы применимости метода RSFI

## Где метод работает
✅ Эмбеддеры размерности ≥768d (лучше ≥1024d)
✅ Few-shot режим (≤5 размеченных примеров): конкурентоспособен с LogReg
✅ Стилистически разнородные данные (шаблонные атаки vs чистые промпты)
✅ Первый рубеж многослойной защиты (быстрая предфильтрация)

## Где метод НЕ работает / работает хуже
❌ Однородные распределения (разнообразные unsafe vs разнообразные safe из одного источника): ожидаемая просадка до 0.6-0.7 AUC
❌ При N_ref ≥ 10: LogReg на тех же данных стабильно лучше на 2.5-5.3 п.п.
❌ Низкоразмерные эмбеддеры (384d): ZCA вредит, метод проигрывает наивному косинусу
❌ Адаптивные атаки (GCG, obfuscation): latent embedding collapse → AUC ~0.6-0.7

## Рекомендуемый режим использования
- Эмбеддер: BAAI/bge-large-en-v1.5 (1024d) или Qwen3-Embedding (4096d)
- N_ref: 50-100 атак (баланс между качеством и стоимостью разметки)
- Без ZCA на моделях <1024d, с ZCA на ≥4096d
- В составе multi-layer defense (RSFI → LLM-judge → human review)
```

---

## Исправления кода (1 день)

### `src/benchmarks/run_fitted_subspace_sweep.py`

```diff
- calib_corpus = safe_texts[:800]  # УТЕЧКА: включает тестовые safe
+ calib_corpus = safe_texts[ref_safe]  # Только референсная выборка

- best_k = max(auc_by_k, key=auc_by_k.get)  # Выбор k по ТЕСТУ
+ # Выбор k по валидации
+ auc_val = {k: roc_auc_score(y_val, scores_val[k]) for k in K_LIST}
+ best_k = max(auc_val, key=auc_val.get)
+ # Репорт на тесте с этим k
+ final_auc = roc_auc_score(y_test, scores_test[best_k])
```

### `src/benchmarks/sota_benchmark_suite.py`

```diff
- logreg.fit(E[:int(0.3*len(E))], y[:int(0.3*len(E))])
- auc = roc_auc_score(y, logreg.decision_function(E))  # Оценка на трейне!
+ logreg.fit(E[train_idx], y[train_idx])
+ auc = roc_auc_score(y[test_idx], logreg.decision_function(E[test_idx]))

- cov = np.cov(E.T)  # Вся выборка
+ cov = np.cov(E[train_idx].T)  # Только трейн
```

### Cleanup

```bash
# Удалить / переместить в legacy/
mkdir -p data/results/legacy
mv data/results/harmbench_sfi_results.csv data/results/legacy/
mv data/results/sfi_balanced_800_results.csv data/results/legacy/
mv src/benchmarks/run_jbb_experiment.py src/benchmarks/legacy/
```

---

## Финальная статья: структура

### 1. Введение
- Проблема: jailbreaks, prompt injection, сикофантия
- Существующие методы: slow (Llama Guard 100-200ms) или high FPR (cosine)
- Наш подход: геометрическая фильтрация через SVD-подпространство

### 2. Метод
- Формулы: ZCA, log-map, SVD-разложение (из math.md, секции 2.1-2.5)
- Алгоритм: калибровка на N_ref примерах, скоринг в O(kd)

### 3. Эксперименты
- Датасеты: TrustAIRLab (2000), ToxicChat (2853), XSTest (450)
- Модели: mpnet-768d, bge-1024d, Qwen3-4096d
- Протокол: honest holdout, k по валидации, 10 сидов, DeLong-тест
- Бейзлайны: naive cosine, LogReg, Prompt-Guard-86M, ProtectAI deberta

### 4. Результаты
**Таблица 1: Мультимодельные результаты**
| Эмбеддер | d | RSFI | Cosine | LogReg |
|----------|---|------|--------|--------|
| mpnet | 768 | 0.821 | 0.785 | 0.874 |
| bge | 1024 | 0.838 | 0.801 | 0.882 |
| Qwen3 | 4096 | **0.856** | 0.797 | 0.881 |

**Таблица 2: Операционная точка**
| Метод | TPR@FPR≤1% | TPR@FPR≤0.1% | PR-AUC |
|-------|------------|--------------|--------|
| (результаты E3) |

**Таблица 3: Head-to-head внешние бейзлайны**
| (результаты E7) |

**График 1:** AUC vs N_ref (few-shot ниша)  
**График 2:** AUC vs k (SVD-подпространство)  
**График 3:** Вклад компонент (ablation: raw / +SVD / +ZCA)

### 5. Анализ
- **Где метод работает:** 768d+, few-shot ≤5, стилистически разнородные данные
- **Где проигрывает:** однородные данные, N_ref ≥ 10 (LogReg лучше), адаптивные атаки
- **Ablation:** SVD даёт +4-6 п.п., ZCA полезен только на 4096d

### 6. Границы применимости
- Честно признать просадку на ToxicChat/XSTest (если подтвердится)
- Обфусцированные атаки — открытая проблема для всех геометрических методов

### 7. Заключение
- Метод подходит как быстрый первый рубеж (few-shot, низкая латентность)
- Не заменяет LLM-judge, но дополняет (multi-layer defense)
- Future work: адаптивные атаки, мультимодальность (image jailbreaks)

---

## Чеклист перед подачей

### Эксперименты
- [ ] E2: ToxicChat + XSTest прогнаны, результаты в CSV
- [ ] E3: TPR@FPR≤1%/0.1% + PR-AUC репортятся
- [ ] E7: Head-to-head таблица на общих данных
- [ ] E5: Whitening stability на 4096d (опционально, но желательно)
- [ ] E10: DeLong-тест, bootstrap CI, 10 сидов

### Код
- [ ] Все утечки убраны (whitening только на ref, k по val)
- [ ] Фейковые датасеты в `legacy/`
- [ ] `honest_eval_final.py` — финальный скрипт, воспроизводящий всё

### Документация
- [ ] README: AUC 0.856, latency ~10-15ms, few-shot (не zero-shot)
- [ ] math.md: пороги унифицированы, честные цифры
- [ ] benchmarks.md: HarmBench 100% удалён
- [ ] limitations.md: добавлен раздел "Где метод НЕ работает"

### Статья
- [ ] Структура по шаблону выше
- [ ] Все таблицы заполнены реальными данными
- [ ] Графики сгенерированы (`matplotlib`, высокое качество)
- [ ] Честное сравнение с бейзлайнами
- [ ] Раздел Limitations заполнен

---

## Ожидаемый исход

### Оптимистичный сценарий
✅ E2 (ToxicChat/XSTest): RSFI просядет до 0.75-0.78, но МЕНЬШЕ косинуса (0.68-0.72) → вклад подтверждён  
✅ E7: RSFI в топ-3 среди fast methods (после Prompt-Guard, но лучше k-NN)  
✅ Статья принята на workshop (NeurIPS SoLaR, ACL SRW, ICLR TinyPapers) или К3-сборник  
✅ Диплом защищён с формулировкой "исследование границ применимости геометрических методов"

### Реалистичный сценарий
⚠️ E2: RSFI просядет вровень с косинусом (оба ~0.70) → вклад SVD нивелируется на однородных данных  
⚠️ E7: RSFI в середине таблицы  
⚠️ Статья на workshop со скидкой на negative result ("двухрежимность геометрических фильтров")  
✅ Диплом защищён, но без публикации в индексируемом журнале

### Пессимистичный сценарий
❌ E2: RSFI проигрывает косинусу на однородных данных  
❌ E7: LogReg/Prompt-Guard значительно лучше  
⚠️ Переформулировка: "анализ провала геометрических методов на однородных распределениях" → короткая заметка или техрепорт  
✅ Диплом защищён как "исследовательская работа с отрицательным результатом"

---

**Главное:** Negative result, честно измеренный и объяснённый, ценнее inflated SOTA. Научное сообщество уважает честность больше, чем завышенные цифры.
