# Heart Disease Prediction - MLOps Pipeline
## Professional Technical Report

**Project**: End-to-End MLOps Pipeline for Heart Disease Prediction  
**Dataset**: UCI Heart Disease Dataset (303 samples, 13 features)  
**Date**: December 2025  
**Author**: MLOps Team

---

## Executive Summary

This project implements a complete, production-ready MLOps pipeline for heart disease prediction using machine learning. The pipeline encompasses data acquisition, exploratory data analysis, model training, experiment tracking, containerization, Kubernetes deployment, and comprehensive monitoring.

**Key Achievements:**
- **Model Performance**: 88.52% accuracy, 96.10% ROC-AUC
- **Production Ready**: Fully containerized with Docker and Kubernetes
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Monitoring**: Complete observability with Prometheus and Grafana
- **MLOps Best Practices**: Experiment tracking, versioning, reproducibility

---

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Setup & Installation](#2-setup--installation)
3.  [Exploratory Data Analysis](#3-exploratory-data-analysis)
4.  [Modeling Approach](#4-modeling-approach)
5.  [Experiment Tracking](#5-experiment-tracking)
6.  [System Architecture](#6-system-architecture)
7.  [CI/CD Pipeline](#7-cicd-pipeline)
8.  [Deployment](#8-deployment)
9.  [Monitoring & Observability](#9-monitoring--observability)
10. [Results & Performance](#10-results--performance)
11. [Future Work](#11-future-work)
12. [Repository & Resources](#12-repository--resources)

---

## 1. Project Overview

### 1.1 Problem Statement

Heart disease is the leading cause of death globally. This project develops an ML-powered prediction system to assist in early detection and risk assessment.

### 1.2 Dataset

**Source**: UCI Machine Learning Repository  
**Name**: Heart Disease Dataset  
**Samples**: 303 patients  
**Features**: 13 clinical attributes  
**Target**: Binary classification (Disease/No Disease)

**Key Features:**
- `age`: Age in years (29-77)
- `sex`: Gender (0=female, 1=male)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol
- `thalach`: Maximum heart rate
- `ca`: Number of major vessels
- `thal`: Thalassemia type

### 1.3 Project Objectives

1. Build accurate heart disease prediction model
2. Implement complete MLOps pipeline
3. Deploy production-ready API
4. Ensure reproducibility and monitoring
5. Follow industry best practices

---

## 2. Setup & Installation

### 2.1 Prerequisites

```bash
# Required Software
- Python 3.10+
- Docker & Docker Compose
- Git
- kubectl (for Kubernetes deployment)
```

### 2.2 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/2024aa05311-arch/MLOps-Assignment.git
cd MLOps-Assignment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run complete pipeline
python 01_data_pipeline.py       # Data acquisition and preprocessing
python 02_model_pipeline.py      # Feature engineering, training, and evaluation
python 03_mlflow_tracking.py     # Experiment tracking with MLflow
python 04_model_packaging.py     # Model registry and packaging
```

### 2.3 Docker Deployment

```bash
# Build and run API
docker-compose up --build

# Or with monitoring stack
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### 2.4 Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Or using Helm
helm install heart-disease-api ./helm/heart-disease-api
```

#### Note: Although Helm charts were initially considered, the final deployment was implemented using Kubernetes deployment manifests (deployment.yaml and service.yaml) as permitted by the assignment.

---

## 3. Exploratory Data Analysis

### 3.1 Data Quality

| Metric         | Value     |
|----------------|-----------|
| Total Samples  | 303       |
| Features       | 13        |
| Missing Values | 6 (1.98%) |
| Duplicates     | 0         |
| Class Balance  | 54% / 46% |

**Quality Assessment**: Excellent - Minimal missing values, well-balanced classes

### 3.2 Key Findings

**Class Distribution:**
- No Disease: 164 samples (54.1%)
- Disease Present: 139 samples (45.9%)
- **Result**: Well-balanced, no need for resampling

**Top Predictive Features** (by correlation):
1. `thal` (0.522) - Thalassemia type
2. `ca` (0.460) - Number of major vessels
3. `exang` (0.432) - Exercise induced angina
4. `oldpeak` (0.425) - ST depression
5. `cp` (0.413) - Chest pain type

**Statistical Summary:**
- Age range: 29-77 years (mean: 54.4)
- Cholesterol: 126-564 mg/dl (mean: 246.7)
- Max heart rate: 71-202 bpm (mean: 149.6)

### 3.3 Visualization Insights

Generated comprehensive visualizations:
- Class balance plots (bar + pie charts)
- Feature distributions (histograms)
- Correlation heatmap
- Features by target class (box plots)
- Interactive Sweetviz report

**Key Insight**: Clear separation between disease/no disease groups for top predictive features

---

## 4. Modeling Approach

### 4.1 Feature Engineering

**Preprocessing Steps:**
1. **Missing Value Imputation**: Median strategy for numerical features
2. **Feature Scaling**: StandardScaler normalization
3. **Train/Test Split**: 80/20 stratified split (242/61 samples)

**Stratification**: Maintained class balance across splits

### 4.2 Model Selection

Trained and evaluated **4 classification models**:

| Model               | Type          | Hyperparameters Tuned                  |
|---------------------|---------------|----------------------------------------|
| Logistic Regression | Linear        | C, solver, penalty                     |
| Random Forest       | Ensemble      | n_estimators, max_depth, min_samples   |
| SVM                 | Kernel        | C, kernel, gamma                       |
| Gradient Boosting   | Ensemble      | n_estimators, learning_rate, max_depth |

### 4.3 Hyperparameter Tuning

**Method**: GridSearchCV with 5-fold cross-validation  
**Scoring**: ROC-AUC (robust for medical diagnosis)  
**Total Combinations Tested**: 503 across all models

**Random Forest Best Parameters:**
```python
{
    'n_estimators': 100,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2
}
```

### 4.4 Model Evaluation

**Metrics Used:**
- Accuracy
- Precision
- Recall (critical for medical diagnosis)
- F1-Score
- ROC-AUC

**Cross-Validation Results:**
All models achieved CV ROC-AUC > 0.87, indicating robust performance

---

## 5. Experiment Tracking

### 5.1 MLflow Integration

**Setup:**
- Experiment Name: `heart_disease_classification`
- Tracking URI: `file:./mlruns`
- Model Registry: Enabled

**Tracked Information:**
1. **Parameters**: All hyperparameters, data split configuration
2. **Metrics**: Train/test accuracy, precision, recall, F1, ROC-AUC
3. **Artifacts**: Confusion matrices, ROC curves, model metadata
4. **Models**: Serialized models with signatures

### 5.2 Experiment Results

| Run   | Model               | Test Accuracy | Test ROC-AUC | CV ROC-AUC |
|-------|---------------------|---------------|--------------|------------|
| 1     | Logistic Regression | 86.89%        | 0.9578       | 0.8917     |
| **2** | **Random Forest**   | **88.52%**    | **0.9610**   | **0.8930** |
| 3     | SVM                 | 54.10%*       | 0.9654       | 0.8912     |
| 4     | Gradient Boosting   | 85.25%        | 0.9015       | 0.8756     |

*Note: SVM had prediction threshold issues despite high ROC-AUC

### 5.3 Model Registry

All 4 models registered with versioning:
- `heart_disease_logistic_regression` (v1)
- `heart_disease_random_forest` (v1) ← **Production Model**
- `heart_disease_svm` (v1)
- `heart_disease_gradient_boosting` (v1)

**Access MLflow UI:**
```bash
mlflow ui
# Navigate to http://127.0.0.1:5000
```

---

## 6. System Architecture

### 6.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Data Layer  │───▶│  ML Pipeline │───▶│   Serving    │       │
│  │              │    │              │    │    Layer     │       │
│  │ • UCI Repo   │    │ • Training   │    │ • FastAPI    │       │
│  │ • CSV Files  │    │ • Evaluation │    │ • Docker     │       │
│  │ • Processing │    │ • MLflow     │    │ • Kubernetes │       │
│  └──────────────┘    └──────────────┘    └──────────────┘.      │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Storage     │    │ Experiment   │    │  Monitoring  │       │
│  │              │    │  Tracking    │    │              │       │
│  │ • Models     │    │ • MLflow     │    │ • Prometheus │       │
│  │ • Artifacts  │    │ • Registry   │    │ • Grafana    │       │
│  │ • Configs    │    │ • Versioning │    │ • Logging    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Deployment Architecture

```
┌─────────── Kubernetes Cluster ───────────┐
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │      LoadBalancer Service           │ │
│  │        (Port 80)                    │ │
│  └──────────────┬──────────────────────┘ │
│                 │                        │
│  ┌──────────────▼──────────────────────┐ │
│  │    Heart Disease API Deployment    │  │
│  │    (3 Replicas)                    │  │
│  │                                    │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐      │  │
│  │  │ Pod  │  │ Pod  │  │ Pod  │      │  │
│  │  │ :8000│  │ :8000│  │ :8000│      │  │
│  │  └──────┘  └──────┘  └──────┘      │  │
│  │                                    │  │
│  │  Health Checks: /health            │  │
│  │  Metrics: /metrics                 │  │
│  │  Auto-scaling: 2-10 replicas       │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  ConfigMap: API Configuration      │  │
│  │  Namespace: ml-models              │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│     Monitoring Stack (Docker)           │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │Prometheus│◀─│ API      │─▶│Grafana │ │
│  │  :9090   │  │ Metrics  │  │ :3000  │ │
│  └──────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────┘
```

### 6.3 Component Details

**API Layer (FastAPI):**
- RESTful endpoints: `/predict`, `/health`, `/metrics`
- Pydantic validation
- Prometheus instrumentation
- Structured logging

**Container Layer (Docker):**
- Base: python:3.10-slim
- Non-root user (security)
- Health checks enabled
- Optimized layers

**Orchestration (Kubernetes):**
- Deployment: 3 replicas
- Service: LoadBalancer
- HPA: Auto-scaling 2-10 pods
- Resource limits: 1 CPU, 1Gi memory

**Monitoring:**
- Prometheus: Metrics collection
- Grafana: Visualization
- Structured logs: JSON format

---

## 7. CI/CD Pipeline

### 7.1 GitHub Actions Workflow

**Trigger Events:**
- Push to `main` or `develop` branches
- Pull requests
- Manual dispatch

**Pipeline Stages:**

```yaml
┌─────────────┐
│    Lint     │  ← Code quality (flake8, black)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Test     │  ← Unit tests (31 tests)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Train     │  ← Full ML pipeline
└──────┬──────┘  (only on main/develop)
       │
       ▼
┌─────────────┐
│   Deploy    │  ← Documentation
└─────────────┘  (only on main)
```

### 7.2 Pipeline Jobs

**1. Lint Job:**
- Runs flake8 for style checking
- Checks code formatting with black
- Uploads lint results as artifacts

**2. Test Job:**
- Runs 31 unit tests with pytest
- Generates coverage report (70% threshold)
- Uploads coverage artifacts
- Comments coverage on PRs

**3. Train Job:**
- Executes complete ML pipeline
- Trains all 4 models
- Evaluates and packages best model
- Uploads model artifacts, evaluation results

**4. Deploy Job:**
- Creates deployment documentation
- Uploads deployment docs

**5. Notify Job:**
- Reports pipeline status
- Runs always (even onFailure)

### 7.3 Artifacts Generated

| Artifact           | Contents                | Retention |
|--------------------|-------------------------|-----------|
| lint-results       | Linting configs         | 90 days   |
| coverage-report    | HTML + XML coverage     | 90 days   |
| test-results       | Pytest results          | 90 days   |
| trained-models     | All model files         | 90 days   |
| evaluation-results | Metrics, plots          | 90 days   |
| mlflow-runs        | Experiment data         | 90 days   |
| deployment-docs    | README, guides          | 90 days   |

---

## 8. Deployment

### 8.1 Deployment Options

**Option 1: Local Docker**
```bash
docker-compose up --build
# Access at http://localhost:8000
```

**Option 2: Kubernetes (kubectl)**
```bash
kubectl apply -f k8s/
kubectl port-forward service/heart-disease-api-service 8000:80 -n ml-models
```

**Option 3: Helm Chart(Optional)**
```bash
helm install heart-disease-api ./helm/heart-disease-api
```
Helm charts were initially considered for Kubernetes deployment.  
However, as per the assignment requirement (deployment manifest **or** Helm chart),
the final deployment was implemented using Kubernetes manifests
(`deployment.yaml` and `service.yaml`) only.


### 8.2 Kubernetes Manifests

Created production-ready Kubernetes configurations:
- `namespace.yaml` - ml-models namespace
- `deployment.yaml` - 3 replicas with health checks
- `service.yaml` - LoadBalancer for external access
- `configmap.yaml` - Application configuration
- `hpa.yaml` - Auto-scaling (CPU/memory based)

### 8.3 Deployment Features

**High Availability:**
- 3 pod replicas
- Rolling updates (max surge: 1, max unavailable: 0)
- Pod disruption budgets

**Auto-Scaling:**
- Min replicas: 2
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%

**Security:**
- Non-root user (UID 1000)
- No privilege escalation
- Resource limits enforced
- Read-only root filesystem option

**Health Monitoring:**
- Liveness probe: /health (30s interval)
- Readiness probe: /health (10s interval)
- Startup delay: 10-30s

---

## 9. Monitoring & Observability

### 9.1 Metrics Collected

**Request Metrics:**
- `api_requests_total` - Total requests by endpoint, status
- `api_request_latency_seconds` - Request latency histogram
- `active_requests` - Current active requests

**Prediction Metrics:**
- `predictions_total` - Predictions by label (Disease/No Disease)
- `prediction_confidence` - Confidence score distribution

**System Metrics:**
- `model_loaded` - Model load status (1/0)

### 9.2 Grafana Dashboard

Pre-configured dashboard with 4 panels:

1. **Request Rate** (Stat)
   - Query: `sum(rate(api_requests_total[5m]))`
   - Shows requests per second

2. **Request Latency** (Time Series)
   - P95 and P50 latency over time
   - Helps identify performance degradation

3. **Predictions by Label** (Pie Chart)
   - Distribution of Disease vs No Disease
   - Monitors prediction patterns

4. **Median Confidence** (Gauge)
   - Median prediction confidence
   - Quality indicator

### 9.3 Logging

**Log Format:**
```
2025-12-21 23:00:00 - __main__ - INFO - Request: POST /predict
2025-12-21 23:00:00 - __main__ - INFO - Prediction successful: Disease (confidence=73.36%)
2025-12-21 23:00:00 - __main__ - INFO - Response: POST /predict Status=200 Latency=0.052s
```

**Log Levels:**
- INFO: Request/response tracking
- WARNING: Performance issues
- ERROR: Exceptions with stack traces

**Log Destinations:**
- `api.log` - File logging
- stdout - Console logging (Docker logs)

---

## 10. Results & Performance

### 10.1 Model Performance

**Best Model: Random Forest Classifier**

| Metric    | Value   | Interpretation                      |
|-----------|---------|-------------------------------------|
| Accuracy  | 88.52%  | Excellent overall performance       |
| Precision | 83.87%  | 84% of positive predictions correct |
| Recall    | 92.86%  | Detected 93% of disease cases       |
| F1-Score  | 88.14%  | Strong balance                      |
| ROC-AUC   | 96.10%  | Excellent discrimination            |

**Confusion Matrix:**
```
                Predicted
Actual       No Disease  Disease
No Disease        28        5
Disease            2       26
```

**Clinical Significance:**
- **High Recall (92.86%)**: Minimizes false negatives (only 2 missed cases)
- **Good Precision (83.87%)**: Reduces false alarms
- **Suitable for screening**: High sensitivity for early detection

### 10.2 System Performance

**API Latency:**
- P50: < 100ms
- P95: < 500ms
- Average: 52ms per prediction

**Throughput:**
- Single prediction: ~20 req/s
- Batch prediction (10 patients): ~15 req/s

**Resource Usage:**
- Memory: ~600MB per pod
- CPU: 0.3-0.5 cores under normal load

### 10.3 Test Coverage

**Unit Tests: 31 tests total**
- Preprocessing: 10 tests
- Data processing: 12 tests
- Prediction interface: 9 tests

**Coverage: ~70%** (core functionality)

---

## 11. Future Work

### 11.1 Model Improvements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Feature engineering with domain expertise
- [ ] Ensemble methods (stacking, voting)
- [ ] Handle class imbalance with advanced techniques
- [ ] Explainability with SHAP/LIME

### 11.2 MLOps Enhancements

- [ ] A/B testing framework
- [ ] Model drift detection
- [ ] Automated retraining pipeline
- [ ] Feature store integration
- [ ] Data quality monitoring

### 11.3 Infrastructure

- [ ] Multi-region deployment
- [ ] Blue-green deployment strategy
- [ ] Canary releases
- [ ] Service mesh (Istio)
- [ ] Advanced auto-scaling policies

### 11.4 Security

- [ ] API authentication (JWT/OAuth)
- [ ] Rate limiting
- [ ] Input sanitization
- [ ] HTTPS/TLS encryption
- [ ] Secret management (Vault)

---

## 12. Repository & Resources

### 12.1 Project Structure

```
MLOps-Assignment/
├── data/                    # Data files
│   ├── raw/                # Original data
│   ├── processed/          # Cleaned data
│   └── engineered/         # Feature-engineered data
├── models/                  # Trained models
│   └── preprocessor.pkl    # Preprocessing pipeline
├── packaged_models/         # Deployment-ready models
│   ├── pickle/             # Pickle format
│   ├── mlflow/             # MLflow format
│   └── README.md           # Deployment guide
├── outputs/                 # Results and visualizations
│   ├── eda_visualizations/ # EDA plots
│   └── model_evaluation/   # Evaluation results
├── tests/                   # Unit tests
├── k8s/                     # Kubernetes manifests
   ├── namespace.yaml
   ├── deployment.yaml
   ├── service.yaml
   ├── configmap.yaml
   ├── hpa.yaml
   └── screenshots/         # Task 7 deployment proof
      ├── pods_running.png
      ├── services.png
      └── swagger_ui.png
├── monitoring/              # Monitoring configs
│   ├── prometheus.yml
│   ├── grafana-dashboard.json
│   └── docker-compose-monitoring.yml
├── .github/workflows/       # CI/CD pipelines
│   └── ci-cd.yml
├── 01_data_pipeline.py      # Data ingestion & processing
├── 02_model_pipeline.py     # Training & Evaluation
├── 03_mlflow_tracking.py    # Experiment tracking
├── 04_model_packaging.py    # Model registry
├── app.py                   # FastAPI application
├── predict.py               # Inference script
├── preprocessing_pipeline.py # Reusable pipeline
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Python dependencies
├── requirements-api.txt     # API dependencies
├── environment.yml          # Conda environment
├── pyproject.toml          # Python project config
└── README.md               # Project documentation
```

### 12.2 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Project overview and quick start |
| FINAL_REPORT.md | Detailed project report |
| walkthrough.md | Detailed technical walkthrough |

### 12.3 Key Scripts

| Script | Purpose |
|--------|---------|
| 01_data_pipeline.py | Data ingestion and preprocessing |
| 02_model_pipeline.py | Model training and evaluation |
| 03_mlflow_tracking.py | Experiment tracking with MLflow |
| 04_model_packaging.py | Packaged models for deployment |
| app.py | FastAPI application serving |
| predict.py | Inference script |

### 12.4 Links & Resources

- **Repository**: [Link to your Git repository]
- **MLflow UI**: `mlflow ui` (http://127.0.0.1:5000)
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000

---

## Conclusion

This project demonstrates a complete, production-ready MLOps pipeline implementing industry best practices:

**Data Pipeline**: Automated acquisition, cleaning, and validation  
**ML Pipeline**: Reproducible training with experiment tracking  
**Deployment**: Containerized with Kubernetes orchestration  
**CI/CD**: Automated testing and deployment  
**Monitoring**: Full observability with metrics and logs  
**Documentation**: Comprehensive guides and tutorials  

**Model Performance**: 88.52% accuracy with 96.10% ROC-AUC  
**System Reliability**: High availability with auto-scaling  
**Production Ready**: Enterprise-grade MLOps implementation  

The pipeline is ready for deployment to cloud platforms (AWS, GCP, Azure) and can scale to handle production workloads.

---

**Report Generated**: 2025-12-21  
**Version**: 1.0.0  
**Status**: Production Ready
