# 🛡️ Fraud Detection System (End-to-End ML Pipeline)

A full-scale fraud detection pipeline built to identify and respond to fraudulent transactions using CatBoost, drift monitoring, and a FastAPI-based inference service. Designed with production-readiness and MLOps extensibility in mind.

---

## 🌟 Project Objective

Detect fraudulent credit card transactions with high recall while minimizing false positives. Given the severe class imbalance and business impact of missed frauds, the model is optimized primarily for **Recall** and **PR-AUC**.

---

## 🧱 Project Architecture

```
                        ┌────────────────────────────────────┐
                        │   Raw Transaction Data │
                        └──────────────────────────────┘
                                   │
                             Preprocessing
                                   ↓
                   ┌───────────────────────────────┐
                   │ Feature Engineering Module │
                   └───────────────────────────────┘
                                   ↓
                ┌──────────────────────────────────────────────┐
                │ CatBoost / XGBoost / RandomForest  │
                └───────────────────────────────┐
                                 │
                    ┌────────────────────────────┐
                    │ Model Evaluation (CV)   │
                    │ Metrics: PR-AUC, Recall │
                    └────────────────────────┘
                                 ↓
                        ┌──────────────────────────┐
                        │ Model Registry  │
                        │    (MLflow)     │
                        └────────────────────────┘
                                 ↓
                 ┌────────────────────────────────────────────────┐
                 │ FastAPI Inference REST API   │
                 └────────────────────────────────────────────────┘
                                ↓
          ┌────────────────────────────────────────────────┐
          │ Monitoring & Drift Detection (hourly)     │
          └────────────────────────────────────────────────┘
```

---

## ⚙️ Key Components

| Module               | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `train.py`           | Initial model training with evaluation and saving to disk & MLflow  |
| `predict.py`         | Batch inference on samples or CSV data                              |
| `api/main.py`        | FastAPI app serving fraud predictions                               |
| `retrain.py`         | Full retraining logic on new labeled data                           |
| `metric_control.py`  | Scheduled monitoring of model quality and data/concept drift        |
| `retrain_starter.py` | Automatically decides if retrain is needed based on threshold rules |
| `src/`               | Modular code for data loading, feature engineering, model training  |
| `tests/`             | Unit tests for logic and REST API endpoints                         |

---

## 📊 Core Metrics & Monitoring

### Model Quality Metrics:

* **Recall**: sensitivity to frauds
* **PR-AUC**: precision-recall trade-off
* **Accuracy**: (logged, but not optimized for)

### Drift Detection:

* **Data Drift**: Kolmogorov–Smirnov tests on features between baseline and current data
* **Concept Drift**: appearance of new features in incoming data not seen during training

### Thresholds:

* Retrain triggered if:

  * Recall drops by ≥ 5% vs. baseline
  * PR-AUC drops by ≥ 5%
  * > 3 features show significant data drift
  * Concept drift detected (new unseen features)

---

## 🧪 Testing

Run all tests using:

```bash
pytest tests/
```

Covers:

* `predict.py`
* API routes
* Feature engineering
* Drift metrics

---

## 🚀 Deployment

### FastAPI (development):

```bash
uvicorn api.main:app --reload
```

Access Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Inference Example:

```bash
POST /predict
[
  {
    "Time": 10000,
    "Amount": 42.00,
    "V1": -1.23,
    "V2": 0.99
  }
]
```

---

## 🔁 Retraining Cycle

### Automated retraining:

```bash
python metric_control.py        # logs metrics hourly (cron/airflow ready)
python retrain_starter.py      # checks metrics_log.csv, triggers retrain
```

### Manual retraining:

```bash
python retrain.py
```

---

## 📈 Future Enhancements

* [ ] Slack / Email alerts for failed retrain
* [ ] Integration with Grafana via DB or Prometheus
* [ ] Containerization with Docker
* [ ] Cron / Airflow scheduling
* [ ] MLflow Model Registry & promotion logic
* [ ] Streaming or real-time data ingestion

---

## 🧠 Author

**Alekseev Stanis**
ML/DS Infrastructure, Systemic ML Engineering, Production-Ready Workflows
GitHub: [@AlekseevStanis](https://github.com/AlekseevStanis)

---

## 📄 License

MIT License — see `LICENSE.md` if included.

```
```
