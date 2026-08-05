<div align="center">
  
# 🛡️ RSFI: Riemannian System Fidelity Index 🌐

<p align="center">
  <a href="README_EN.md"><strong>🇺🇸 English Version</strong></a> &nbsp; | &nbsp; <a href="README_RU.md"><strong>🇷🇺 Русская Версия</strong></a>
</p>

<p align="center">
  <b>A Method for Dynamic Control of Semantic Drift and Sycophancy in Large Language Models<br>Based on Non-Euclidean Geometry and ZCA Whitening</b>
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


## Результаты независимых экспериментов (E2-E10)
Недавний запуск расширенного набора экспериментов показал следующие ограничения и сильные стороны метода RSFI:

1. **Гомогенные наборы данных (E2):** На датасетах со схожей стилистикой (например, ToxicChat) метрика ROC-AUC для RSFI-SVD падает до 0.668, уступая даже простому косинусному расстоянию (0.927). Метод сильно зависит от стилистического разнообразия.
2. **Адаптивные атаки и обфускация (E6):** При использовании base64, rot13 и других обфускаций атаки коллапсируют в чистое многообразие в пространстве эмбеддингов. ROC-AUC для RSFI падает до 0.16. Это фундаментальное ограничение всех геометрических методов без участия LLM.
3. **Строгие операционные точки (E3):** Несмотря на более низкий общий ROC-AUC, при строгих ограничениях на False Positive Rate (FPR = 1% или 0.1%), RSFI-SVD сохраняет больше детекций (TPR), чем обычный косинус, что делает его полезным для zero-shot фильтров первой линии.
4. **Сравнение с внешними бейзлайнами (E7):** Тяжелые NLP-модели (например, ProtectAI-deberta) работают медленно (8-9 мс). RSFI-SVD работает за 0.46 мс и не требует хранения больших индексов баз данных (как k-NN, который дает 0.80 AUC), оставаясь быстрым легковесным решением.
5. **Статистическая значимость (E10):** Bootstrap-тестирование с поправками Холма и тест Делонга (p < 0.0001) подтвердили статистически значимую разницу: supervised-методы (LogReg) стабильно лучше наивного косинуса, который, в свою очередь, в среднем обходит RSFI по чистому AUC.
6. **Стабильность отбеливания (E5):** Эксперимент завершается. ZCA-отбеливание показывает ограничения при нехватке калибровочных данных.
