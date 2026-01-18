# 🚀 MLOps Production Pipeline
## � Problem Statement

### Objective
Build a machine learning model to predict whether a bank customer will subscribe to a term deposit based on their demographics, banking history, and campaign interaction data.

### Dataset
- **Source**: [Kaggle Playground Series - Season 5, Episode 8](https://www.kaggle.com/competitions/playground-series-s5e8)
- **Type**: Synthetically generated data (similar to UCI Bank Marketing dataset)
- **Training Samples**: 45,211 records
- **Features**: 16 features (numerical + categorical)
- **Target**: Binary classification (subscription: yes/no)

### Business Context
The bank aims to optimize its marketing campaign efficiency by identifying customers most likely to subscribe to term deposits. This reduces marketing costs and improves campaign ROI by targeting high-probability prospects.

### Success Metrics
- **Primary**: ROC-AUC Score ≥ 0.95
- **Secondary**: F1-Score ≥ 0.90
- **Tertiary**: Model latency < 100ms for API inference

### Task
**Binary Classification**: Predict subscription probability for each customer with probability estimates for threshold tuning.

---
<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-3.8.1-0194E2?style=for-the-badge&logo=mlflow)
![Airflow](https://img.shields.io/badge/Airflow-3.0.6-017CEE?style=for-the-badge&logo=apache-airflow)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2_|_ECR-FF9900?style=for-the-badge&logo=amazon-aws)
![DVC](https://img.shields.io/badge/DVC-Enabled-13ADC7?style=for-the-badge&logo=dvc)

**Enterprise-grade ML pipeline with automated training, deployment, and monitoring**

[Features](#-features) • [Architecture](#-architecture) • [Performance](#-model-performance) • [Quick Start](#-quick-start) • [Deployment](#-deployment)

</div>

---

## 📊 Model Performance

<div align="center">

| Model | AUC-ROC | Accuracy | F1-Score | 
|-------|---------|----------|----------|
| **LightGBM** | **0.97** | 94.2% | 0.93 |
| **XGBoost** | 0.96 | 93.8% | 0.92 | 
| **PyTorch FNN** | 0.95 | 93.1% | 0.91 | 

</div>

### 📈 ROC Curves & Confusion Matrices

All models tracked with **MLflow** including:
- ROC/AUC curves for each model
- Confusion matrices with precision/recall
- Feature importance visualizations
- Hyperparameter tuning history

---

## 📸 Screenshots

### MLflow Experiment Tracking
![MLflow UI](screenshot/mlflow.png)
*MLflow dashboard showing model metrics, parameters, and experiment history*

### Airflow DAG Orchestration
![Airflow DAG](screenshot/Airflow.png)
*Airflow DAG visualization with 6-task pipeline and execution logs*

### GitHub Actions CI/CD
![GitHub Actions](screenshot/cicd.png)
*Automated CI/CD pipeline with lint, test, build, and deployment stages*

---

## ✨ Features

### 🔄 **End-to-End ML Pipeline**
- **Data Versioning** with DVC for reproducible datasets
- **Experiment Tracking** with MLflow (metrics, parameters, artifacts)
- **Workflow Orchestration** with Apache Airflow for scheduled retraining
- **CI/CD Pipeline** with GitHub Actions (lint, test, build, deploy)

### 🤖 **Multiple ML Algorithms**
- **XGBoost**: Gradient boosting (50 estimators, optimized)
- **LightGBM**: Fast gradient boosting (baseline)
- **PyTorch Neural Network**: 3-layer FNN with dropout & early stopping
- **Hyperparameter Tuning**: Optuna-based optimization

### 🐳 **Production Ready**
- **Docker**: Containerized Flask application
- **AWS EC2**: Scalable compute instances
- **AWS ECR**: Private Docker registry
- **GitHub Actions**: Self-hosted runners for CI/CD

### 🎯 **Deployment**
- **Flask REST API** for real-time predictions
- **Model serving** with automatic versioning
- **Health checks** and monitoring endpoints

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Raw Data] -->|DVC| B[Data Ingestion]
    B --> C[Data Transformation]
    C --> D[Model Training]
    D -->|XGBoost| E[MLflow Tracking]
    D -->|LightGBM| E
    D -->|PyTorch FNN| E
    E --> F[Model Registry]
    F --> G[Flask API]
    G -->|Docker| H[AWS ECR]
    H -->|Deploy| I[AWS EC2]
    
    J[Airflow] -.->|Schedule| D
    K[GitHub Actions] -.->|CI/CD| H
```

### 📦 Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Data Pipeline** | Ingestion, validation, transformation | Pandas, Scikit-learn |
| **Model Training** | Train & evaluate ML models | XGBoost, LightGBM, PyTorch |
| **Experiment Tracking** | Log metrics, params, artifacts | MLflow 3.8.1 |
| **Orchestration** | Schedule & monitor pipelines | Apache Airflow 3.0.6 |
| **Version Control** | Track data & model versions | DVC + Git |
| **API Server** | Serve predictions via REST | Flask |
| **Containerization** | Package application | Docker |
| **Registry** | Store Docker images | AWS ECR |
| **Compute** | Host production services | AWS EC2 |
| **CI/CD** | Automated testing & deployment | GitHub Actions |

---



## �🚀 Quick Start

### Prerequisites

```bash
# Python 3.14+
python --version

# Docker installed
docker --version

# AWS CLI configured
aws --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/Amit95688/ml-ops.git
cd ml-ops

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc pull  # Download tracked data
```

### Run Locally

#### 1️⃣ **Start MLflow UI**
```bash
python scripts/launch_mlflow_ui.py
# Access at: http://localhost:5000
```

#### 2️⃣ **Start Airflow**
```bash
AIRFLOW_HOME=$(pwd)/airflow python scripts/start_airflow.py
# Access at: http://localhost:8080
# Login: admin / YHKfeaptbhGrBCke
```

#### 3️⃣ **Train Models**
```bash
python src/pipeline/train_pipeline.py
# Results logged to MLflow automatically
```

#### 4️⃣ **Launch Flask API**
```bash
python application.py
# API running at: http://localhost:5000
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t ml-ops-api:latest .
```

### Run Container

```bash
docker run -d \
  -p 5000:5000 \
  --name ml-ops-api \
  ml-ops-api:latest
```

### Docker Compose (MLflow + Airflow)

```bash
# Start MLflow UI and Airflow (API server + scheduler)
docker compose up -d mlflow airflow

# Stop everything
docker compose down
```

- MLflow UI: http://localhost:5000
- Airflow UI: http://localhost:8080 (local dev auth is disabled; admin/admin also works)
- DAGs mount the local `dags/` folder; MLflow artifacts persist in the `mlflow-artifacts` volume.

Airflow orchestration view (sample):
![Airflow DAG](screenshot/Airflow.png)

### Push to AWS ECR

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag ml-ops-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-ops-api:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-ops-api:latest
```

---

## ☁️ AWS Deployment

### EC2 Setup

```bash
# SSH to EC2 instance
ssh -i key.pem ec2-user@<public-ip>

# Pull from ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-ops-api:latest

# Run on EC2
docker run -d -p 80:5000 \
  --restart always \
  --name ml-ops-api \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-ops-api:latest
```

### Security Groups
- Inbound: Port 80 (HTTP), 443 (HTTPS)
- Outbound: All traffic

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: MLOps CI/CD

on:
  push:
    branches: [main]

jobs:
  continuous-integration:
    runs-on: ubuntu-latest
    steps:
      - Lint & format check
      - Run unit tests
      - Data validation
      
  continuous-delivery:
    needs: continuous-integration
    steps:
      - Build Docker image
      - Push to AWS ECR
      
  continuous-deployment:
    needs: continuous-delivery
    runs-on: self-hosted  # AWS EC2 runner
    steps:
      - Pull latest image
      - Deploy to production
      - Health check validation
```

### Self-Hosted Runner on EC2

```bash
# Configure GitHub Actions runner on EC2
./config.sh --url https://github.com/Amit95688/ml-ops \
  --token <YOUR_TOKEN>

./run.sh
```

---

## 📊 MLflow Integration

### Tracking Server

```bash
# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### Log Experiments

```python
import mlflow

with mlflow.start_run(run_name="XGBoost_Training"):
    mlflow.log_params({"n_estimators": 50, "max_depth": 5})
    mlflow.log_metric("auc_roc", 0.97)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")
```

---

## 🔧 DVC Pipeline

### Pipeline Stages

```bash
# View pipeline DAG
dvc dag

# Reproduce entire pipeline
dvc repro

# Push data to remote storage
dvc push

# Pull data from remote
dvc pull
```

### Pipeline Definition (`dvc.yaml`)

```yaml
stages:
  data_ingestion:
    cmd: python -m src.components.data_ingestion
    outs:
      - artifacts/train.csv
      - artifacts/test.csv
      
  model_trainer:
    cmd: python src/pipeline/train_pipeline.py
    deps:
      - artifacts/train.csv
    outs:
      - artifacts/model.pkl
```

---

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Predict
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [5.1, 3.5, 1.4, 0.2, ...]
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.97,
  "model": "LightGBM",
  "version": "v1.2.3"
}
```

---

## 📁 Project Structure

```
ml-ops/
├── .github/
│   └── workflows/
│       └── workflow.yml          # CI/CD pipeline
├── artifacts/                    # Model artifacts & data
├── dags/
│   └── ml_training_pipeline.py  # Airflow DAG
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── pytorch_model.py
│   └── pipeline/
│       ├── train_pipeline.py
│       └── predict_pipeline.py
├── scripts/
│   ├── start_airflow.py
│   └── launch_mlflow_ui.py
├── tests/                        # Unit & integration tests
├── application.py                # Flask API
├── Dockerfile                    # Container definition
├── dvc.yaml                      # DVC pipeline
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Integration tests
pytest tests/integration/
```

---

## 📈 Monitoring

### Airflow Dashboard
- **URL**: `http://localhost:8080`
- Monitor DAG runs, task status, execution logs
- Schedule automatic retraining (weekly)

### MLflow UI
- **URL**: `http://localhost:5000`
- Compare model versions
- View experiment metrics & artifacts
- Download trained models

### Application Logs
```bash
tail -f logs/app_$(date +%Y-%m-%d).log
```

---

## 🛠️ Configuration

### Environment Variables

```bash
# .env file
AWS_REGION=us-east-1
ECR_REPOSITORY=ml-ops-api
MLFLOW_TRACKING_URI=http://localhost:5000
AIRFLOW_HOME=/path/to/airflow
```

### Model Parameters

Edit `dvc.yaml` to tune hyperparameters:
```yaml
params:
  xgboost:
    n_estimators: 100
    max_depth: 7
  lightgbm:
    num_leaves: 50
```

---

## 🤝 Contributing

```bash
# Fork the repository
# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Open Pull Request
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👨‍💻 Author

**Amit Dubey**

[![GitHub](https://img.shields.io/badge/GitHub-Amit95688-181717?style=for-the-badge&logo=github)](https://github.com/Amit95688)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/amit-dubey-45292629a/)

---

## 🙏 Acknowledgments

- MLflow for experiment tracking
- Apache Airflow for orchestration
- DVC for data versioning
- AWS for cloud infrastructure
- GitHub Actions for CI/CD automation

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the ML community

</div>
