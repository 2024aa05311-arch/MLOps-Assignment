# Heart Disease Prediction - Complete MLOps Pipeline

**End-to-End ML Model with CI/CD and Production Deployment**

---

## 🎯 Quick Overview

- **Problem**: Predict heart disease risk from patient health data
- **Dataset**: UCI Heart Disease (303 samples, 13 features)
- **Model**: Random Forest - 88.52% accuracy, 96.10% ROC-AUC
- **Tech Stack**: Python, FastAPI, Docker, Kubernetes, MLflow, Prometheus/Grafana
- **Status**: ✅ Production Ready

---

## 🚀 Quick Start (3 Steps)

### 1. Setup
```bash
# Clone and install
git clone <your-repo-url>
cd MLOps-Assignment
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
# Complete ML pipeline (one command) — updated consolidated scripts
python 01_data_pipeline.py && \
python 02_model_pipeline.py && \
python 03_mlflow_tracking.py && \
python 04_model_packaging.py
```

### 3. Deploy
```bash
# Option A: Docker (Easiest)
docker-compose up -d

# Option B: Kubernetes
kubectl apply -f k8s/

# Option C: Local
python app.py
```

**Access**: http://localhost:8000/docs

---

## 📁 Simple Project Structure

```
MLOps-Assignment/
├── 01_data_pipeline.py          # Data acquisition & preprocessing pipeline
├── 02_model_pipeline.py         # Feature engineering, training, evaluation pipeline
├── 03_mlflow_tracking.py        # Track experiments with MLflow
├── 04_model_packaging.py        # Package best model for serving
├── app.py                       # FastAPI application
├── predict.py                   # Inference script / client usage examples
├── preprocessing_pipeline.py    # Reusable preprocessing pipeline (class)
├── packaged_models/             # Packaged model artifacts (pickle / mlflow)
├── Dockerfile                   # Container config
├── docker-compose.yml           # Docker setup
├── requirements.txt             # Dependencies
├── requirements-api.txt         # API-specific dependencies
├── .github/workflows/ci-cd.yml  # CI/CD pipeline
├── k8s/                         # Kubernetes configs
├── helm/                        # Helm chart
├── tests/                       # Unit tests
├── monitoring/                  # Prometheus/Grafana
└── README.md                    # This file
```

---

## 💡 Key Features

✅ **Complete ML Pipeline** - Data → Training → Evaluation → Deployment  
✅ **Experiment Tracking** - MLflow for all experiments  
✅ **CI/CD Pipeline** - GitHub Actions with tests  
✅ **Containerization** - Docker & Kubernetes ready  
✅ **Auto-Scaling** - Kubernetes HPA (2-10 pods)  
✅ **Monitoring** - Prometheus + Grafana dashboards  
✅ **Testing** - 31 unit tests, 70% coverage  
✅ **Production Ready** - FastAPI with health checks  

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Model | Random Forest |
| Accuracy | 88.52% |
| Precision | 83.87% |
| Recall | 92.86% |
| F1-Score | 88.14% |
| ROC-AUC | 96.10% |

**Clinical Significance**: Only 2 missed disease cases out of 28 (High recall for screening)

---

## 🔧 Detailed Setup

### Prerequisites
- Python 3.10+
- Docker (for containerization)
- kubectl (for Kubernetes)
- Git

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Or use Conda
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## 🎓 Pipeline Explanation

### 1. Data Pipeline
```bash
# Step 1: Get data from UCI repository
python 01_data_acquisition.py

# Step 2: Clean and preprocess
python 02_preprocessing.py

# Step 3: Exploratory data analysis
python 03_eda.py
```

**Output**: Clean dataset + visualizations in `outputs/eda_visualizations/`

### 2. ML Pipeline
```bash
# Step 4: Feature engineering (scaling, splitting)
python 04_feature_engineering.py

# Step 5: Train 4 models with GridSearchCV
python 05_model_training.py

# Step 6: Evaluate and compare models
python 06_model_evaluation.py
```

**Models Trained**: Logistic Regression, Random Forest, SVM, Gradient Boosting  
**Output**: Trained models in `models/`, results in `outputs/model_evaluation/`

### 3. Experiment Tracking
```bash
# Step 7: Log everything to MLflow
python 07_mlflow_tracking.py

# View experiments
mlflow ui  # http://localhost:5000
```

**Tracked**: Parameters, metrics, artifacts, model registry

### 4. Model Packaging
```bash
# Step 8: Package best model
python 09_model_packaging.py
```

**Output**: Pickle + MLflow formats in `packaged_models/`

---

## 🐳 Deployment Options

### Option 1: Docker (Recommended)
```bash
# Build and run
docker-compose up -d

# Test
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Stop
docker-compose down
```

### Option 2: Kubernetes
```bash
# Deploy all resources
kubectl apply -f k8s/

# Check status
kubectl get all -n ml-models

# Access API (port forward)
kubectl port-forward service/heart-disease-api-service 8000:80 -n ml-models

# Clean up
kubectl delete namespace ml-models
```

### Option 3: Helm Chart
```bash
# Install
helm install heart-disease-api ./helm/heart-disease-api

# Upgrade
helm upgrade heart-disease-api ./helm/heart-disease-api

# Uninstall
helm uninstall heart-disease-api
```

### Option 4: Local Development
```bash
# Run directly
python app.py

# Or with uvicorn
uvicorn app:app --reload
```

---

## 📈 Monitoring Stack

Start Prometheus + Grafana:
```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

**Access Points**:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**Metrics Tracked**:
- Request rate & latency
- Prediction counts
- Model confidence
- System health

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# View coverage
open htmlcov/index.html

# Linting
flake8 .
black --check .
```

**31 Tests Covering**:
- Preprocessing pipeline (10 tests)
- Data processing (12 tests)
- Prediction interface (9 tests)

---

## 🔄 CI/CD Pipeline

**GitHub Actions** automatically runs on every push/PR:

1. **Lint** - Code quality checks (flake8, black)
2. **Test** - Run 31 unit tests with coverage
3. **Train** - Execute full ML pipeline
4. **Deploy** - Create deployment artifacts

**View**: `.github/workflows/ci-cd.yml`

---

## 📚 API Documentation

### Endpoints

**Health Check**
```bash
GET /health
```

**Prediction (Single)**
```bash
POST /predict
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
  "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
  "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

**Prediction (Batch)**
```bash
POST /predict/batch
{
  "patients": [<patient1>, <patient2>, ...]
}
```

**Metrics**
```bash
GET /metrics  # Prometheus format
```

**Interactive Docs**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🏗️ Architecture

```
┌─────────────┐
│  Data (UCI) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  ML Pipeline    │  ← 7 Python scripts
│  (Training)     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  MLflow         │  ← Experiment tracking
│  (Versioning)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  FastAPI        │  ← REST API
│  (Serving)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Docker/K8s     │  ← Deployment
│  (Production)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Prometheus     │  ← Monitoring
│  + Grafana      │
└─────────────────┘
```

---

## 📝 Assignment Deliverables

✅ **Code**: All ML pipeline + API scripts  
✅ **Docker**: Dockerfile + docker-compose.yml  
✅ **Dependencies**: requirements.txt + environment.yml  
✅ **Dataset**: Download script (01_data_acquisition.py)  
✅ **Notebooks/Scripts**: EDA, training, inference  
✅ **Tests**: tests/ folder with 31 unit tests  
✅ **CI/CD**: GitHub Actions workflow  
✅ **Deployment**: Kubernetes manifests + Helm chart  
✅ **Screenshots**: screenshots/ folder (to be populated)  
✅ **Report**: FINAL_REPORT.md (10+ pages)  

---

## 🎬 Video Demo

[Record video showing]:
1. Run ML pipeline
2. View MLflow experiments
3. Test API with Swagger
4. Deploy with Docker
5. Show Kubernetes deployment
6. View Grafana dashboard

---

## 🔗 Links & Resources

- **Repository**: [Your GitHub URL]
- **Dataset**: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Full Report**: See `FINAL_REPORT.md`
- **Technical Details**: See `walkthrough.md` artifact

---

## 🤝 Contributing

This is an academic project. For production use:
- Add authentication to API
- Implement model drift detection
- Add more comprehensive tests
- Set up proper secrets management
- Configure production logging

---

## 📄 License

[Your License]

---

## 👤 Author

**Name**: [Your Name]  
**Email**: [Your Email]  
**Course**: MLOps Assignment  
**Date**: December 2025  

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready

For detailed technical documentation, see `FINAL_REPORT.md`
