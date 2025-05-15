# 🕵️‍♂️ Fraud Detection System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-enabled-brightgreen)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)
![Status](https://img.shields.io/badge/status-active-success)

> Продакшен-пайплайн для детектирования мошеннических транзакций.  
> Включает обучение, инференс, API, контроль дрифта и retrain-логику.

---

## 📦 Структура проекта

.
├── api/ # FastAPI API
├── data/ # исходные и сырые данные
├── models/ # сохранённые модели (.cbm)
├── src/ # логика обучения, загрузки, фичей
├── tests/ # модульные тесты
├── train.py # первичное обучение модели
├── predict.py # консольный предикт
├── retrain.py # ручной retrain
├── metric_control.py # мониторинг качества и дрифта
├── retrain_starter.py # принимает решение о retrain
├── requirements.txt
└── README.md