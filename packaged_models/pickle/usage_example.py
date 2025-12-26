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