"""
Unit Tests for Data Processing
Tests data acquisition and preprocessing functions
"""

import pytest
import pandas as pd
import numpy as np
import os


class TestDataAcquisition:
    """Test suite for data acquisition"""

    def test_raw_data_exists(self):
        """Test if raw data file exists"""
        assert os.path.exists("data/raw/heart_disease_raw.csv")

    def test_raw_data_structure(self):
        """Test raw data structure"""
        df = pd.read_csv("data/raw/heart_disease_raw.csv")

        # Check shape
        assert len(df) > 0
        assert len(df.columns) >= 13

        # Check expected columns
        expected_cols = [
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
        for col in expected_cols:
            assert col in df.columns

    def test_raw_data_types(self):
        """Test data types in raw data"""
        df = pd.read_csv("data/raw/heart_disease_raw.csv")

        # Numerical columns should be numeric
        numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        for col in numerical_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])


class TestDataPreprocessing:
    """Test suite for data preprocessing"""

    def test_processed_data_exists(self):
        """Test if processed data exists"""
        assert os.path.exists("data/processed/heart_disease_processed.csv")

    def test_processed_data_no_missing(self):
        """Test processed data has no missing values"""
        df = pd.read_csv("data/processed/heart_disease_processed.csv")

        # Should have no missing values
        assert df.isnull().sum().sum() == 0

    def test_processed_data_target(self):
        """Test processed data has target variable"""
        df = pd.read_csv("data/processed/heart_disease_processed.csv")

        # Should have target_binary column
        assert "target_binary" in df.columns

        # Target should be binary (0 or 1)
        unique_values = df["target_binary"].unique()
        assert set(unique_values).issubset({0, 1})

    def test_engineered_data_exists(self):
        """Test if engineered data exists"""
        assert os.path.exists("data/engineered/train_data.csv")
        assert os.path.exists("data/engineered/test_data.csv")

    def test_train_test_split(self):
        """Test train/test split ratio"""
        train_df = pd.read_csv("data/engineered/train_data.csv")
        test_df = pd.read_csv("data/engineered/test_data.csv")

        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total

        # Should be approximately 20% test (0.2 ± 0.05)
        assert 0.15 < test_ratio < 0.25

    def test_scaled_features(self):
        """Test that features are scaled"""
        train_df = pd.read_csv("data/engineered/train_data.csv")

        # Remove target column
        if "target" in train_df.columns:
            features = train_df.drop("target", axis=1)
        else:
            features = train_df

        # Mean should be close to 0
        mean_vals = features.mean()
        assert abs(mean_vals.mean()) < 0.5, "Features not centered"

        # Std should be close to 1
        std_vals = features.std()
        assert 0.5 < std_vals.mean() < 1.5, "Features not scaled properly"


class TestDataQuality:
    """Test suite for data quality checks"""

    def test_no_duplicates(self):
        """Test for duplicate rows"""
        df = pd.read_csv("data/processed/heart_disease_processed.csv")

        duplicates = df.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate rows"

    def test_value_ranges(self):
        """Test value ranges for key features"""
        df = pd.read_csv("data/raw/heart_disease_raw.csv")

        # Age should be reasonable
        assert df["age"].min() >= 0
        assert df["age"].max() <= 120

        # Sex should be binary
        if df["sex"].notna().any():
            assert set(df["sex"].dropna().unique()).issubset({0, 1})

        # Blood pressure should be positive
        assert df["trestbps"].min() > 0
        assert df["trestbps"].max() < 300

    def test_class_balance(self):
        """Test class balance in processed data"""
        df = pd.read_csv("data/processed/heart_disease_processed.csv")

        if "target_binary" in df.columns:
            value_counts = df["target_binary"].value_counts()

            # Neither class should be less than 20% of total
            min_class_pct = value_counts.min() / len(df)
            assert min_class_pct > 0.2, "Severe class imbalance detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
