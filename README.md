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
[![Few-Shot Protection](https://img.shields.io/badge/Security-Few--Shot%20Black--Box-brightgreen.svg)]()
[![Latency Sub-Millisecond](https://img.shields.io/badge/Latency-%3C%2010%20ms%20(Real--Time)-orange.svg)]()
[![Open Science](https://img.shields.io/badge/Open%20Science-Validated-blueviolet.svg)]()

<br>

*This repository contains the official implementation, benchmarks, telemetry data, and mathematical proofs for the RSFI algorithm.*<br>
*Please choose your preferred language above to read the full documentation.*

</div>


## Ключевые результаты (все числа воспроизводимы из `data/results/*.csv`)

Полная методология и таблицы — в [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md). Протокол: leakage-free holdout, 5 сидов, DeLong-тесты.

1. **Одноклассовые геометрические фильтры теряют качество из-за игнорирования безопасного класса.** Дискриминантное направление $\mu_{mal}-\mu_{safe}$ (метод $B1$, хранит ровно 1 вектор, O(d) на запрос) обходит наивный косинус на гетерогенном Wild на +7.7…+10.5 п.п. ROC-AUC (5/5 сидов, p < 0.0001, DeLong; `data/results/E2d_delong_tests.csv`).
2. **Отбеливание помогает не всегда, и это объяснимо.** Через разложение $\Sigma_T = \Sigma_W + \Sigma_B$: на омонимическом XSTest отбеливание даёт +13.6 п.п. ($0.762 \to 0.897$), а на гетерогенном Wild — вредит, так как сжимает ранговое собственное значение вдоль разделяющего направления. Отбеливание по внутриклассовой ковариации $\Sigma_W^{-1/2}$ восстанавливает часть потерь (`data/results/E8_sigma_w.csv`).
3. **Ограничения зафиксированы честно:** адаптивные обфускации (base64, rot13) выводят атаки из геометрического многообразия (AUC падает до ~0.16, `data/results/E6_adaptive_attacks.csv`) — фундаментальное ограничение всех эмбеддинговых фильтров без участия LLM; supervised LogReg на том же бюджете в среднем сильнее $B1$ по чистому AUC.
4. **Рекомендуемая роль:** ультрабыстрый first-line фильтр в архитектуре defense-in-depth перед тяжёлыми LLM-судьями, а не замена им.
