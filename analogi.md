# Глава: Обзор аналогов, конкурентных решений и смежных исследований

## 1. Таксономия подходов к обеспечению безопасности и детекции дрейфа LLM

Современные методы защиты языковых моделей от контекстного дрейфа, сикофантии и состязательных атак (Jailbreak / Prompt Injection) можно систематизировать по нескольким ключевым классификационным признакам.

**По схеме доступа к модели:**
- **White-Box (Белый ящик):** Методы, требующие прямого доступа к внутренним активациям, весам или градиентам модели (Representation Engineering, Abliteration, TrajGuard, CALM, ORBA).
- **Black-Box (Чёрный ящик):** Методы, работающие исключительно на уровне ввода-вывода (API) без доступа к внутренностям модели (Llama Guard, NeMo Guardrails, ZEDD, SAFENUDGE, Lakera Guard, OpenAI Moderation API).

**По уровню анализа:**
- **Token-level:** Детекция на уровне отдельных токенов (SafeDecoding, ShieldHead).
- **Vector-level / Embedding-level:** Анализ эмбеддингов предложений или токенов (ZEDD, SAFENUDGE, Centroid-based Guardrails).
- **Sentence-level / Utterance-level:** Детекция на уровне полных высказываний (Llama Guard, ConSol).
- **Trajectory-level:** Мониторинг динамики скрытых состояний в процессе декодирования (TrajGuard).

**По режиму обучения:**
- **Training-based (Обучаемые):** Методы, требующие обучения или дообучения сторонних классификаторов (Llama Guard 1–4, NeMo Guardrails, SAFENUDGE, Guardrails AI).
- **Zero-Shot / Training-Free:** Методы, не требующие дополнительного обучения (ZEDD, RepE-интервенции, TrajGuard, RSFI).

**По математической модели пространства:**
- **Евклидова геометрия:** Косинусное сходство, евклидовы расстояния, PCA (большинство современных методов).
- **Риманова геометрия:** Методы, учитывающие кривизну и неевклидову структуру латентного пространства (RSFI, Riemannian Mean Pooling, Hyperbolic classifiers).

***

## 2. Подробный аналитический разбор категорий аналогов

### 2.1. Методы «Белого ящика» (Representation Engineering, Abliteration, TrajGuard, CALM, ORBA)

**Representation Engineering (RepE)** — направление, систематизированное в работе Zou et al. (2023, arXiv:2310.01405), предлагающее подход к прозрачности ИИ через анализ популяционных представлений rather than отдельных нейронов. RepE позволяет выявлять «направления концептов» (concept directions) в пространстве активаций и манипулировать ими для контроля поведения модели. Для задач безопасности RepE используется для выделения «направления безопасности» (safety steering direction) через PCA на разностных векторах между безопасными и вредными инструкциями. [arxiv](https://arxiv.org/abs/2310.01405)

**Abliteration** — техника «разъединения» моделей через слияние (model merging) с ортогонализацией весов относительно «направления отказа» (refusal direction), описанная в работах Hammoud et al. (2024) и Labonne (2024). Метод позволяет удалять встроенные механизмы отказа (refusal mechanisms) из моделей или, наоборот, усиливать их. Ключевое ограничение: требует прямого доступа к весам и активациям модели, что делает метод неприменимым к закрытым API (GPT-4, Claude, Gemini). [aclanthology](https://aclanthology.org/2024.findings-emnlp.762.pdf)

**Refusal Direction (Arditi et al., 2024, arXiv:2406.11717)** — фундаментальное исследование, показавшее, что отказ в 13 популярных чат-моделях (до 72B параметров) опосредован одномерным подпространством. Удаление этого направления из residual stream предотвращает отказ на вредные инструкции, а добавление — вызывает отказ даже на безвредные. Это доказывает хрупкость текущих методов safety fine-tuning, но также указывает на возможность точечных интервенций. Ограничение: требует white-box доступа и не защищает от адаптивных атак, нацеленных на подавление этого направления. [arxiv](https://arxiv.org/abs/2406.11717)

**TrajGuard (Liu et al., 2026, ACL Findings)** — метод детекции jailbreak-атак через мониторинг траекторий скрытых состояний в процессе декодирования. TrajGuard использует Streaming Geometric Surveillance (SGS) для отслеживания отклонений hidden states от безопасного региона в латентном пространстве с последующей семантической верификацией через PAIR-Judge. Метод достигает 95% защиты при задержке 5.2 мс/токен и FPR < 1.5%. Ограничение: требует доступа к hidden states (неприменим к black-box API), а также калибровки референсных распределений на доменных данных. [aclanthology](https://aclanthology.org/2026.findings-acl.655/)

**ORBA (Orthogonal Reflection Bounded Ablation, 2026)** — метод ортогональной абляции через отражение Хаусхолдера, обеспечивающий сохранение нормы активаций и семантическую стабильность. ORBA обобщает abliteration, RepE и PEFT через единый формализм направленной абляции. Ограничение: white-box доступ, необходимость выделения направления интервенции. [huggingface](https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation)

**Фундаментальные ограничения White-Box методов:**
1. Неприменимость к закрытым API (OpenAI, Anthropic, Google).
2. Требование доступа к активациям/весам, что исключает развёртывание в production-средах с проприетарными моделями.
3. Уязвимость к адаптивным атакам, нацеленным на обход конкретных направлений интервенции.
4. Высокая вычислительная стоимость мониторинга в реальном времени.

***

### 2.2. Внешние нейросетевые классификаторы и дискриминаторы (Llama Guard 1–4, NeMo Guardrails, Lakera, Guardrails AI)

**Llama Guard (Meta AI, 2023–2025)** — семейство моделей-гардов на базе Llama (8B–12B параметров), предназначенных для классификации входных и выходных данных LLM на предмет безопасности. Llama Guard 3 (2024) достигает 93.9% точности на бенчмарках, но требует генерации полного ответа перед модерацией, что вносит задержку 38–43 мс на запрос. Llama Guard 4 (12B, 2025) — мультимодальная модель, способная анализировать текст и изображения одновременно, но с ещё большими вычислительными затратами. [huggingface](https://huggingface.co/meta-llama/Llama-Guard-4-12B)

**NVIDIA NeMo Guardrails (2024–2026)** — фреймворк для создания guardrails на уровне диалоговых потоков, поддерживающий детекцию prompt injection, jailbreak, PII leakage и токсичности. NeMo Guardrails использует комбинацию правил (dialog flows) и нейросетевых классификаторов (LLM-based classifiers). Ограничение: задержка >100 мс из-за необходимости запуска отдельной LLM для классификации. [reddit](https://www.reddit.com/r/ollama/comments/1f4iwv0/protecting_against_prompt_injection/)

**Lakera Guard (2025–2026)** — API для защиты LLM-приложений от prompt injection, jailbreak и утечек данных в реальном времени. Lakera поддерживает детекцию по 4 категориям (prompt attacks, data leakage, content violation, unknown links) и интеграцию через единый REST-эндпоинт. Ограничение: облачная зависимость, задержка сети, невозможность кастомизации под специфические домены. [appsecsanta](https://appsecsanta.com/lakera)

**Guardrails AI (2024–2026)** — Python-фреймворк с компонуемой архитектурой валидаторов для проверки выводов LLM на токсичность, PII, формат, галлюцинации. Guardrails AI поддерживает интеграцию с различными LLM и предоставляет extensible validator ecosystem. Ограничение: требует запуска дополнительных классификаторов, высокая задержка. [veto](https://veto.so/compare/ai-guardrails)

**OpenAI Moderation API (2023–2025)** — встроенный инструмент для проверки контента на соответствие политикам использования OpenAI. Поддерживает детекцию по категориям: violence, self-harm, sexual content, hate speech. С моделью omni-moderation-latest (на базе GPT-4o) API анализирует текст и изображения. Ограничение: работает только для OpenAI API, задержка сети, ограниченная кастомизация. [community.openai](https://community.openai.com/t/is-the-moderation-api-required/330419)

**Фундаментальные ограничения нейросетевых гардов:**
1. **Высокая задержка (Latency > 100–200 мс):** необходимость запуска отдельной LLM для классификации каждого запроса/ответа.
2. **Вычислительные затраты:** модели гардов (8B–12B параметров) требуют значительных GPU-ресурсов.
3. **Необходимость переобучения:** гарды требуют постоянного обновления под новые типы атак (zero-day уязвимости).
4. **Ложные срабатывания (FPR 5–15%):** особенно на чувствительных, но безопасных запросах (XSTest бенчмарк).
5. **Пост-фактум детекция:** многие гарды проверяют ответ после генерации, что не предотвращает утечку вредоносного контента.

***

### 2.3. Легковесные фильтры на основе текстовых эмбеддингов (ZEDD, SAFENUDGE, Centroid-based)

**ZEDD (Zero-Shot Embedding Drift Detection, Sekar et al., 2025–2026)** — метод детекции prompt injection через измерение семантического дрейфа в пространстве эмбеддингов. ZEDD формализует детекцию как статистическую гипотезу: сравниваются эмбеддинги подозреваемого промпта и его чистой версии через косинусное сходство. Порог определяется через GMM/KDE на калибровочных данных. Метод достигает 90–95% точности на LLMail-Inject датасете при задержке <50 мс. **Ограничения:** [github](https://github.com/AnirudhSekar/ZEDD)
- Проблема анизотропии: эмбеддинги трансформеров распределены в узком конусе («semantic cone»), что снижает дискриминативность косинусного сходства. [cnrs.hal](https://cnrs.hal.science/hal-04471739/file/ACL_2023_ait-saada.pdf)
- Высокий FPR (2–5%) на минимально изменённых промптах.
- Плоская евклидова геометрия: косинусное сходство игнорирует кривизну латентного пространства.

**SAFENUDGE (Fonseca et al., 2025, EMNLP)** — метод, комбинирующий Controlled Text Generation с «подталкиванием» (nudging) модели к безопасным ответам во время генерации. SAFENUDGE триггерится при детекции jailbreak и перенаправляет декодирование через интервенции в пространстве токенов. Метод снижает успешные jailbreak на 28–37% с минимальной задержкой. **Ограничения:** требует модификации процесса декодирования, работает только с открытыми моделями, не защищает от zero-day атак. [aclanthology](https://aclanthology.org/2025.emnlp-main.1010/)

**Centroid-based Embedding Guardrails (2026)** — ансамблевый метод, сравнивающий входные эмбеддинги с заранее вычисленными центроидами известных атак (pattern-based, cluster-based, task-specific). Использует all-MiniLM-L6-v2 (384-dim) и усреднение топ-3 схожестей. **Ограничения:** [blog.frohrer](https://blog.frohrer.com/multi-path-ensemble-detection-of-prompt-injection-attacks-via-embedding-similarity-trajectory-analysis-and-fine-tuned-classification/)
- Зависимость от полноты библиотеки центроидов (не детектирует zero-day атаки).
- Анизотропия: центроиды в анизотропном пространстве смещены к доминирующим направлениям.
- FPR растёт при увеличении числа центроидов.

**Фундаментальные ограничения embedding-методов:**
1. **Анизотропия (semantic cone):** векторы предложений в трансформерах занимают узкий конус в латентном пространстве, что приводит к высоким косинусным схожестям даже для семантически разных текстов. [cnrs.hal](https://cnrs.hal.science/hal-04471739/file/ACL_2023_ait-saada.pdf)
2. **Отсутствие учёта геометрии:** косинусное сходство предполагает евклидово пространство, игнорируя кривизну и риманову структуру. [academic.oup](https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag220/8726335)
3. **Высокий FPR:** 2–15% на бенчмарках (XSTest, Alpaca).
4. **Одиночный вектор атаки:** большинство методов используют одномерное направление или центроид, что уязвимо к адаптивным атакам.

***

### 2.4. Последовательные статистические и вероятностные методы (ConSol, Wald-SPRT, Sovereign Verification Engine)

**ConSol (Consistency-based LLM Verification, 2025–2026)** — метод оценки согласованности ответов LLM через множественные семплы и статистический анализ. Используется для детекции фактических ошибок и галлюцинаций. [users.ece.cmu](https://users.ece.cmu.edu/~lbauer/papers/2025/emnlp2025-llm-consistency.pdf)

**Wald-SPRT (Sequential Probability Ratio Test)** — классический статистический метод для последовательного тестирования гипотез, применяемый в LLMOps для детекции дрейфа данных. [orq](https://orq.ai/blog/model-vs-data-drift)

**Sovereign Verification Engine (2026)** — система для верификации выводов LLM в некритичных средах через многоступенчатую проверку. [sei.cmu](https://www.sei.cmu.edu/blog/an-approach-to-accelerate-verification-and-software-standards-testing-with-llms/)

**MAKER (Million-Step Zero-Error Reasoning, 2025)** — фреймворк для достижения нулевых ошибок в рассуждениях LLM через декомпозицию и верификацию шагов. [cognizant](https://www.cognizant.com/us/en/ai-lab/blog/maker)

**Отличия от RSFI:**
- Эти методы ориентированы на **экономию токенов** и **верификацию рассуждений**, а не на ортогональное подпространство угроз.
- Не используют риманову геометрию или ZCA-обеливание.
- Не обеспечивают zero-shot детекцию jailbreak на уровне эмбеддингов.

***

## 3. Индустриальные тренды и концепции из независимого исследовательского сообщества (2025–2026 гг.)

**LessWrong / Alignment Forum:**
- **Activation Patching / Causal Tracing:** техника интерпретируемости через замену активаций между различными входами для выявления причинных механизмов. Сообщество активно исследует «refusal direction removal» и «steering vector transfer» для управления поведением моделей. [arxiv](https://arxiv.org/html/2404.15255v1)
- **Abliteration Community Research:** энтузиасты разрабатывают методы «uncensoring» LLM через ортогонализацию весов, что парадоксально подтверждает существование узких направлений отказа. [maximelabonne.substack](https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e)

**Reddit (r/LocalLLaMA, r/MachineLearning):**
- Обсуждения guardrails для сикофантии: пользователи добавляют «sycophancy guardrails» в system prompts для снижения соглашательства. [reddit](https://www.reddit.com/r/LLMDevs/comments/1sjcrnf/llm_sycophancy_with_example/)
- Практические гайды по NeMo Guardrails и prompt injection detection. [reddit](https://www.reddit.com/r/ollama/comments/1f4iwv0/protecting_against_prompt_injection/)

**GitHub Issues / Discussions:**
- Репозиторий `refusal_direction` (Arditi et al.) содержит код для извлечения и применения направления отказа. [github](https://github.com/andyrdt/refusal_direction)
- ZEDD (AnirudhSekar/ZEDD) — open-source реализация zero-shot drift detection. [github](https://github.com/AnirudhSekar/ZEDD)
- TrajGuard (Liu et al.) — репозиторий с кодом для streaming hidden-state monitoring. [aclanthology](https://aclanthology.org/2026.findings-acl.655/)

**Концептуальные идеи сообщества:**
- **Hidden Context Drift:** идея о том, что контекст может «дрейфовать» в латентном пространстве без явных изменений в тексте. [atlan](https://atlan.com/know/context-drift-detection/)
- **Prompt Pollution:** измеряемое расстояние между исходным намерением и текущим контекстом через косинусную схожесть эмбеддингов. [getmaxim](https://www.getmaxim.ai/articles/how-context-drift-impacts-conversational-coherence-in-ai-systems/)
