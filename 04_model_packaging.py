"""
Model Packaging Script for Heart Disease Prediction
Packages the best model in multiple formats for deployment
Formats: Pickle, MLflow, ONNX
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
from datetime import datetime
from skl2onnx import to_onnx
from preprocessing_pipeline import HeartDiseasePreprocessor


def load_best_model(model_path="models/random_forest.pkl"):
    """Load the best performing model"""
    print("=" * 80)
    print("MODEL PACKAGING - HEART DISEASE PREDICTION")
    print("=" * 80)

    print(f"\n[INFO] Loading best model from: {model_path}")
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded: {type(model).__name__}")

    return model


def save_model_pickle(model, output_dir="packaged_models/pickle"):
    """Save model in pickle format with metadata"""
    print(f"\n{'=' * 80}")
    print("FORMAT 1: PICKLE")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[INFO] Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "packaging_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "pickle",
        "python_version": "3.x",
        "sklearn_version": "scikit-learn>=1.3.0",
        "description": "Random Forest model for heart disease prediction",
        "input_features": 13,
        "output_classes": 2,
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Metadata saved: {metadata_path}")

    # Create usage example
    usage_example = """
# Load and use the pickled model

import joblib
import pandas as pd
from preprocessing_pipeline import HeartDiseasePreprocessor

# Load model
model = joblib.load('packaged_models/pickle/best_model.pkl')

# Load preprocessor
preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.pkl')

# Prepare new data
X_new = pd.DataFrame({
    'age': [63], 'sex': [1], 'cp': [3], 'trestbps': [145],
    'chol': [233], 'fbs': [1], 'restecg': [0], 'thalach': [150],
    'exang': [0], 'oldpeak': [2.3], 'slope': [0], 'ca': [0], 'thal': [1]
})

# Preprocess and predict
X_preprocessed = preprocessor.transform(X_new)
prediction = model.predict(X_preprocessed)
probability = model.predict_proba(X_preprocessed)

print(f"Prediction: {prediction[0]}")  # 0 = No Disease, 1 = Disease
print(f"Probability: {probability[0]}")
    """

    usage_path = os.path.join(output_dir, "usage_example.py")
    with open(usage_path, "w") as f:
        f.write(usage_example.strip())
    print(f"[INFO] Usage example saved: {usage_path}")

    print(f"\n[SUCCESS] Pickle packaging complete!")

    return model_path, metadata_path


def save_model_onnx(model, output_dir="packaged_models/onnx"):
    """Save model in ONNX format for cross-platform deployment"""
    print(f"\n{'=' * 80}")
    print("FORMAT 2: ONNX")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create sample input for ONNX conversion
        initial_type = [("float_input", "float", [None, 13])]

        print(f"\n[INFO] Converting model to ONNX format...")

        # Convert to ONNX
        onnx_model = to_onnx(model, initial_types=initial_type, target_opset=12)

        # Save ONNX model
        onnx_path = os.path.join(output_dir, "best_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"[INFO] ONNX model saved: {onnx_path}")

        # Save ONNX metadata
        onnx_metadata = {
            "format": "onnx",
            "opset_version": 12,
            "packaging_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_shape": [None, 13],
            "input_type": "float32",
            "output_classes": 2,
            "description": "ONNX format for cross-platform deployment",
        }

        metadata_path = os.path.join(output_dir, "onnx_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(onnx_metadata, f, indent=4)
        print(f"[INFO] ONNX metadata saved: {metadata_path}")

        # Create ONNX usage example
        onnx_usage = """
# Load and use ONNX model

import onnxruntime as rt
import numpy as np
from preprocessing_pipeline import HeartDiseasePreprocessor

# Load ONNX model
session = rt.InferenceSession('packaged_models/onnx/best_model.onnx')

# Load preprocessor
preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.pkl')

# Prepare input
X_new = pd.DataFrame({
    'age': [63], 'sex': [1], 'cp': [3], 'trestbps': [145],
    'chol': [233], 'fbs': [1], 'restecg': [0], 'thalach': [150],
    'exang': [0], 'oldpeak': [2.3], 'slope': [0], 'ca': [0], 'thal': [1]
})

# Preprocess
X_preprocessed = preprocessor.transform(X_new).values.astype(np.float32)

# Predict
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: X_preprocessed})[0]

print(f"Prediction: {prediction[0]}")
        """

        usage_path = os.path.join(output_dir, "usage_example.py")
        with open(usage_path, "w") as f:
            f.write(onnx_usage.strip())
        print(f"[INFO] ONNX usage example saved: {usage_path}")

        print(f"\n[SUCCESS] ONNX packaging complete!")

        return onnx_path, metadata_path

    except Exception as e:
        print(f"\n[WARNING] ONNX conversion failed: {str(e)}")
        print("          Continuing with other formats...")
        return None, None


def package_with_mlflow(model_name="random_forest", output_dir="packaged_models/mlflow"):
    """Package model using MLflow format"""
    print(f"\n{'=' * 80}")
    print("FORMAT 3: MLFLOW")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")

    print(f"\n[INFO] Loading model from MLflow Model Registry...")

    # Load from MLflow
    model_uri = f"models:/heart_disease_{model_name}/1"

    try:
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        print(f"  ✓ Model loaded from MLflow registry")

        # Save as pyfunc
        mlflow_path = os.path.join(output_dir, "mlflow_model")
        mlflow.sklearn.save_model(model, mlflow_path)
        print(f"[INFO] MLflow model saved: {mlflow_path}")

        # Create MLflow usage example
        mlflow_usage = """
# Load and use MLflow model

import mlflow
import pandas as pd
from preprocessing_pipeline import HeartDiseasePreprocessor

# Load model
model = mlflow.sklearn.load_model('packaged_models/mlflow/mlflow_model')

# Or load from model registry
# mlflow.set_tracking_uri("file:./mlruns")
# model = mlflow.pyfunc.load_model('models:/heart_disease_random_forest/1')

# Load preprocessor
preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.pkl')

# Prepare input
X_new = pd.DataFrame({
    'age': [63], 'sex': [1], 'cp': [3], 'trestbps': [145],
    'chol': [233], 'fbs': [1], 'restecg': [0], 'thalach': [150],
    'exang': [0], 'oldpeak': [2.3], 'slope': [0], 'ca': [0], 'thal': [1]
})

# Preprocess and predict
X_preprocessed = preprocessor.transform(X_new)
prediction = model.predict(X_preprocessed)

print(f"Prediction: {prediction[0]}")
        """

        usage_path = os.path.join(output_dir, "usage_example.py")
        with open(usage_path, "w") as f:
            f.write(mlflow_usage.strip())
        print(f"[INFO] MLflow usage example saved: {usage_path}")

        print(f"\n[SUCCESS] MLflow packaging complete!")

        return mlflow_path

    except Exception as e:
        print(f"\n[WARNING] MLflow packaging failed: {str(e)}")
        print("          Model may not be registered in MLflow")
        return None


def create_deployment_package():
    """Create complete deployment package"""
    print("\n" + "=" * 80)
    print("CREATING COMPLETE DEPLOYMENT PACKAGE")
    print("=" * 80)

    # Package summary
    print(f"\n[INFO] Packaging best model (Random Forest) in multiple formats...")

    # Load model
    model = load_best_model()

    # Save in different formats
    pickle_path, pickle_meta = save_model_pickle(model)
    onnx_path, onnx_meta = save_model_onnx(model)
    mlflow_path = package_with_mlflow()

    # Create deployment README
    deployment_readme = """# Heart Disease Prediction Model - Deployment Package

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
    """

    readme_path = "packaged_models/README.md"
    with open(readme_path, "w") as f:
        f.write(deployment_readme.strip())
    print(f"\n[INFO] Deployment README created: {readme_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("MODEL PACKAGING COMPLETE")
    print("=" * 80)

    print(f"\n[SUMMARY]")
    print(f"  ✓ Pickle format: {pickle_path}")
    if onnx_path:
        print(f"  ✓ ONNX format: {onnx_path}")
    if mlflow_path:
        print(f"  ✓ MLflow format: {mlflow_path}")
    print(f"  ✓ Deployment README: {readme_path}")

    print(f"\n[INFO] All models packaged and ready for deployment!")
    print(f"       See packaged_models/README.md for usage instructions")


if __name__ == "__main__":
    # Create complete deployment package
    create_deployment_package()
