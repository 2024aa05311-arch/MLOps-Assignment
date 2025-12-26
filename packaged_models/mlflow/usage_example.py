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