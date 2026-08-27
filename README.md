<div align="center">
  
# RSFI - Riemannian System Fidelity Index

<p align="center">
  <a href="README_EN.md"><strong>🇺🇸 English Version</strong></a> &nbsp; | &nbsp; <a href="README_RU.md"><strong>🇷🇺 Русская Версия</strong></a>
</p>

<p align="center">
  <b>A Geometric Analysis of One-Class Embedding Guardrails for LLM Jailbreak Detection:<br>Safe-Aware Discriminant Correction and Pooled Within-Class Whitening</b>
</p>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-185%20Passed-brightgreen.svg)]()
[![Latency Sub-Millisecond](https://img.shields.io/badge/Latency-~0.003%20ms%20(GPU)-orange.svg)]()
[![VAK Readiness](https://img.shields.io/badge/VAK%20Readiness-K2%20Verified-blueviolet.svg)]()

<br>

*This repository contains the official implementation, benchmarks, telemetry data, and mathematical proofs for the RSFI algorithm.*<br>
*Please choose your preferred language above to read the full documentation.*

</div>


## О названии

Название «Riemannian System Fidelity Index» — историческое имя проекта. Честная рамка: полный риманов аппарат (логарифмическая/экспоненциальная карты на $\mathbb{S}^{d-1}$) реализован в `src/rsfi/geometry.py` (`RiemannianSphere`) и покрыт юнит-тестами (`tests/test_geometry.py`), однако **бенчмаркированный пайплайн** (все методы в `docs/RESEARCH_REPORT.md`) использует евклидову сферическую геометрию: L2-нормализацию, ZCA-отбеливание Ледуа-Вольфа и SVD/дискриминантные направления. Никаких римановых карт в оценённых методах нет — это осознанное разграничение библиотеки и эксперимента.

## Ключевые результаты (все числа воспроизводимы из `data/results/*.csv`)

Полная методология и таблицы — в [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md). Протокол: leakage-free holdout, 5 сидов, DeLong-тесты.

1. **Одноклассовые геометрические фильтры теряют качество из-за игнорирования безопасного класса.** Дискриминантное направление $\mu_{mal}-\mu_{safe}$ (метод $B1$, хранит ровно 1 вектор, O(d) на запрос) обходит наивный косинус на гетерогенном Wild на +7.7…+10.5 п.п. ROC-AUC (5/5 сидов, p < 0.0001, DeLong; `data/results/E2d_delong_tests.csv`) и превосходит опубликованные внешние детекторы (deberta-v2 0.841, toxic-bert 0.724; `data/results/E9_external_baselines.csv`).
2. **Отбеливание помогает не всегда, и это объяснимо.** Через разложение $\Sigma_T = \Sigma_W + \Sigma_B$: на омонимическом XSTest отбеливание даёт +13.6 п.п. ($0.762 \to 0.897$), а на гетерогенном Wild — вредит, так как сжимает ранговое собственное значение вдоль разделяющего направления. Отбеливание по внутриклассовой ковариации $\Sigma_W^{-1/2}$ восстанавливает в среднем 43.9% потерь на 4 эмбеддерах (`data/results/E8_sigma_w.csv`, `data/results/E8q_qwen_sigma_w.csv`).
3. **Ограничения зафиксированы честно:** под сильной текстовой обфускацией (base64) качество геометрических фильтров деградирует (AUC 0.705 у B1 vs 0.238 у наивного косинуса, `data/results/E6b_obfuscation_boundary.csv`), что требует L1-канонизации (unwrapper) в конвейере.
4. **Рекомендуемая роль:** ультрабыстрый first-line фильтр в архитектуре defense-in-depth перед тяжёлыми LLM-судьями, а не замена им.
