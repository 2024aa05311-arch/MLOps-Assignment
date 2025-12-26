"""
Preprocessing Pipeline for Heart Disease Prediction
Reusable and reproducible preprocessing transformation pipeline
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for Heart Disease dataset

    This transformer handles:
    - Missing value imputation
    - Feature scaling
    - Feature validation

    Usage:
        preprocessor = HeartDiseasePreprocessor()
        X_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    """

    def __init__(self):
        self.feature_names = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.expected_features = [
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

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Training data
        y : array-like, optional
            Target variable (not used, for sklearn compatibility)

        Returns:
        --------
        self
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.expected_features)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Validate features
        self._validate_features(X)

        # Fit imputer
        self.imputer.fit(X)

        # Fit scaler on imputed data
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)

        return self

    def transform(self, X):
        """
        Transform data using fitted pipeline

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to transform

        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed data
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.expected_features)

        # Validate features
        self._validate_features(X)

        # Impute missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed)

        # Convert back to DataFrame
        X_transformed = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        return X_transformed

    def _validate_features(self, X):
        """Validate that input has expected features"""
        if not all(feature in X.columns for feature in self.expected_features):
            missing = [f for f in self.expected_features if f not in X.columns]
            raise ValueError(f"Missing required features: {missing}")

    def get_feature_names(self):
        """Get feature names"""
        return self.feature_names

    def save(self, filepath):
        """Save preprocessor to file"""
        # Ensure the pickled object refers to this module's class name
        # This prevents pickles created when the script was run as
        # __main__ from referencing __main__.HeartDiseasePreprocessor
        # which breaks when loading from another entrypoint (e.g., uvicorn).
        self.__class__.__module__ = "preprocessing_pipeline"
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to: {filepath}")

    @staticmethod
    def load(filepath):
        """Load preprocessor from file.

        Ensures that the `HeartDiseasePreprocessor` class is available on
        `__main__` while unpickling (pytest/uvicorn may run as `__main__`).
        """
        import sys

        main_mod = sys.modules.get("__main__")
        # Temporarily make the class available on __main__ to support
        # pickles that reference __main__.HeartDiseasePreprocessor
        if main_mod is not None:
            setattr(main_mod, "HeartDiseasePreprocessor", HeartDiseasePreprocessor)

        try:
            preprocessor = joblib.load(filepath)
        finally:
            # Clean up the temporary attribute if possible
            try:
                if main_mod is not None and hasattr(main_mod, "HeartDiseasePreprocessor"):
                    delattr(main_mod, "HeartDiseasePreprocessor")
            except Exception:
                pass

        print(f"Preprocessor loaded from: {filepath}")
        return preprocessor


def create_and_save_preprocessor(
    data_path="data/raw/heart_disease_raw.csv", output_path="models/preprocessor.pkl"
):
    """
    Create, fit, and save the preprocessing pipeline

    Parameters:
    -----------
    data_path : str
        Path to raw data
    output_path : str
        Path to save preprocessor

    Returns:
    --------
    preprocessor : HeartDiseasePreprocessor
        Fitted preprocessor
    """
    print("=" * 80)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 80)

    # Load data
    print(f"\n[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Separate features and target
    target_cols = ["num", "target", "target_binary", "condition"]
    feature_cols = [col for col in df.columns if col not in target_cols]

    X = df[feature_cols]

    print(f"  - Features shape: {X.shape}")
    print(f"  - Features: {feature_cols}")

    # Create and fit preprocessor
    print(f"\n[INFO] Creating and fitting preprocessor...")
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(X)

    print(f"  ✓ Preprocessor fitted")

    # Save preprocessor
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preprocessor.save(output_path)

    # Test preprocessing
    print(f"\n[INFO] Testing preprocessing...")
    X_transformed = preprocessor.transform(X)

    print(f"  - Transformed shape: {X_transformed.shape}")
    print(f"  - Mean: {X_transformed.mean().mean():.4f}")
    print(f"  - Std: {X_transformed.std().mean():.4f}")

    print(f"\n{'=' * 80}")
    print("PREPROCESSING PIPELINE CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nPreprocessor saved to: {output_path}")
    print("\nUsage:")
    print("  from preprocessing_pipeline import HeartDiseasePreprocessor")
    print(f"  preprocessor = HeartDiseasePreprocessor.load('{output_path}')")
    print("  X_transformed = preprocessor.transform(X_new)")

    return preprocessor


if __name__ == "__main__":
    # Create and save preprocessor
    preprocessor = create_and_save_preprocessor()
