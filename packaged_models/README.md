# Heart Disease Prediction Model - Deployment Package

## Model Information

- **Model Type**: Random Forest Classifier
- **Performance**: 88.52% Accuracy, 96.10% ROC-AUC
- **Input Features**: 13 clinical attributes
- **Output**: Binary classification (0 = No Disease, 1 = Disease Present)

## Available Formats

### 1. Pickle Format (Python-specific)
- **Location**: `packaged_models/pickle/best_model.pkl`
- **Use Case**: Python applications, quick deployment
- **Requirements**: Python 3.x, scikit-learn
- **Example**: See `packaged_models/pickle/usage_example.py`

### 2. ONNX Format (Cross-platform)
- **Location**: `packaged_models/onnx/best_model.onnx`
- **Use Case**: Production deployment, C++/Java/C# applications
- **Requirements**: ONNX Runtime
- **Example**: See `packaged_models/onnx/usage_example.py`

### 3. MLflow Format (MLOps-ready)
- **Location**: `packaged_models/mlflow/mlflow_model/`
- **Use Case**: MLflow deployments, model serving
- **Requirements**: MLflow
- **Example**: See `packaged_models/mlflow/usage_example.py`

## Preprocessing Pipeline

**Required**: Use the preprocessing pipeline before making predictions

```python
from preprocessing_pipeline import HeartDiseasePreprocessor

preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.pkl')
X_transformed = preprocessor.transform(X_new)
```

## Input Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Sex (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression | Numeric |
| slope | Slope of peak exercise ST | Categorical |
| ca | Number of major vessels (0-3) | Numeric |
| thal | Thalassemia (3, 6, 7) | Categorical |

## Quick Start

```python
import joblib
import pandas as pd
from preprocessing_pipeline import HeartDiseasePreprocessor

# 1. Load model and preprocessor
model = joblib.load('packaged_models/pickle/best_model.pkl')
preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.pkl')

# 2. Prepare input data
patient_data = pd.DataFrame({
    'age': [63], 'sex': [1], 'cp': [3], 'trestbps': [145],
    'chol': [233], 'fbs': [1], 'restecg': [0], 'thalach': [150],
    'exang': [0], 'oldpeak': [2.3], 'slope': [0], 'ca': [0], 'thal': [1]
})

# 3. Preprocess and predict
X_processed = preprocessor.transform(patient_data)
prediction = model.predict(X_processed)[0]
probability = model.predict_proba(X_processed)[0]

# 4. Interpret results
if prediction == 1:
    print(f"Disease detected (Probability: {probability[1]:.2%})")
else:
    print(f"No disease (Probability: {probability[0]:.2%})")
```

## Dependencies

See `requirements.txt` for complete list of dependencies.

## Model Performance

- **Accuracy**: 88.52%
- **Precision**: 83.87%
- **Recall**: 92.86%
- **F1-Score**: 88.14%
- **ROC-AUC**: 96.10%

## Version

- **Model Version**: 1.0
- **Package Date**: {datetime.now().strftime("%Y-%m-%d")}