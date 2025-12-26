"""
Model Inference Script for Heart Disease Prediction
Provides easy-to-use prediction interface with input validation
"""

import joblib
import pandas as pd

from preprocessing_pipeline import HeartDiseasePreprocessor


class HeartDiseasePredictor:
    """
    Prediction interface for heart disease classification

    Usage:
        predictor = HeartDiseasePredictor()
        result = predictor.predict(patient_data)
    """

    def __init__(
        self,
        model_path="packaged_models/pickle/best_model.pkl",
        preprocessor_path="models/preprocessor.pkl",
    ):
        """
        Initialize predictor

        Parameters:
        -----------
        model_path : str
            Path to pickled model
        preprocessor_path : str
            Path to preprocessor
        """
        print("Initializing Heart Disease Predictor...")

        # Load model
        self.model = joblib.load(model_path)
        print(f"  ✓ Model loaded: {type(self.model).__name__}")

        # Load preprocessor
        self.preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)
        print("  ✓ Preprocessor loaded")

        # Define expected features
        self.required_features = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

        print("  ✓ Predictor ready!\n")

    def validate_input(self, data):
        """
        Validate input data

        Parameters:
        -----------
        data : dict or pd.DataFrame
            Patient data

        Returns:
        --------
        pd.DataFrame
            Validated DataFrame
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Check for missing features
        missing_features = [f for f in self.required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Check for extra features
        extra_features = [f for f in data.columns if f not in self.required_features]
        if extra_features:
            print(f"Warning: Ignoring extra features: {extra_features}")
            data = data[self.required_features]

        # Validate data types and ranges
        validations = {
            "age": (29, 77, "Age should be between 29 and 77 years"),
            "sex": (0, 1, "Sex should be 0 (female) or 1 (male)"),
            "cp": (0, 3, "Chest pain type should be 0-3"),
            "trestbps": (94, 200, "Resting blood pressure should be 94-200 mm Hg"),
            "chol": (126, 564, "Cholesterol should be 126-564 mg/dl"),
            "fbs": (0, 1, "Fasting blood sugar should be 0 or 1"),
            "restecg": (0, 2, "Resting ECG should be 0-2"),
            "thalach": (71, 202, "Max heart rate should be 71-202"),
            "exang": (0, 1, "Exercise angina should be 0 or 1"),
            "oldpeak": (0, 6.2, "ST depression should be 0-6.2"),
            "slope": (0, 2, "Slope should be 0-2"),
            "ca": (0, 4, "Number of vessels should be 0-4"),
            "thal": (0, 7, "Thalassemia should be 0-7"),
        }

        for feature, (min_val, max_val, msg) in validations.items():
            if feature in data.columns:
                if data[feature].min() < min_val or data[feature].max() > max_val:
                    print(f"Warning: {msg}")

        return data

    def predict(self, data, return_proba=True):
        """
        Make prediction

        Parameters:
        -----------
        data : dict or pd.DataFrame
            Patient data
        return_proba : bool
            Whether to return probability

        Returns:
        --------
        dict
            Prediction results
        """
        # Validate input
        data_validated = self.validate_input(data)

        # Preprocess
        data_processed = self.preprocessor.transform(data_validated)

        # Predict
        prediction = self.model.predict(data_processed)[0]

        result = {
            "prediction": int(prediction),
            "prediction_label": "Disease" if prediction == 1 else "No Disease",
        }

        if return_proba:
            probabilities = self.model.predict_proba(data_processed)[0]
            result["probability_no_disease"] = float(probabilities[0])
            result["probability_disease"] = float(probabilities[1])
            result["confidence"] = float(max(probabilities))

        return result

    def predict_batch(self, data_list):
        """
        Make predictions for multiple patients

        Parameters:
        -----------
        data_list : list of dict or pd.DataFrame
            Multiple patient data

        Returns:
        --------
        list of dict
            Prediction results for each patient
        """
        if isinstance(data_list, pd.DataFrame):
            # Already a DataFrame — ensure each row is passed as a DataFrame (not Series)
            return [self.predict(data_list.iloc[[i]]) for i in range(len(data_list))]
        else:
            # List of dictionaries
            return [self.predict(data) for data in data_list]


def example_usage():
    """Example usage of the predictor"""
    print("=" * 80)
    print("HEART DISEASE PREDICTOR - EXAMPLE USAGE")
    print("=" * 80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Example patient data
    patient_1 = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    print("\n" + "-" * 80)
    print("SINGLE PREDICTION EXAMPLE")
    print("-" * 80)
    print("\nPatient Data:")
    for key, value in patient_1.items():
        print(f"  {key}: {value}")

    # Make prediction
    result = predictor.predict(patient_1)

    print("\n[PREDICTION RESULT]")
    print(f"  Prediction: {result['prediction_label']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Probability of No Disease: {result['probability_no_disease']*100:.2f}%")
    print(f"  Probability of Disease: {result['probability_disease']*100:.2f}%")

    # Batch prediction example
    patient_2 = {
        "age": 37,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 187,
        "exang": 0,
        "oldpeak": 3.5,
        "slope": 0,
        "ca": 0,
        "thal": 2,
    }

    print("\n" + "-" * 80)
    print("BATCH PREDICTION EXAMPLE")
    print("-" * 80)

    results = predictor.predict_batch([patient_1, patient_2])

    for i, result in enumerate(results, 1):
        print(f"\nPatient {i}:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run example
    example_usage()
