"""
Unit Tests for Preprocessing Pipeline
Tests the HeartDiseasePreprocessor class
"""

import pytest
import pandas as pd
import numpy as np
from preprocessing_pipeline import HeartDiseasePreprocessor


@pytest.fixture
def sample_data():
    """Create sample test data"""
    data = pd.DataFrame(
        {
            "age": [63, 67, 67, 37, 41],
            "sex": [1, 1, 1, 1, 0],
            "cp": [3, 4, 4, 3, 2],
            "trestbps": [145, 160, 120, 130, 130],
            "chol": [233, 286, 229, 250, 204],
            "fbs": [1, 0, 0, 0, 0],
            "restecg": [0, 2, 2, 0, 2],
            "thalach": [150, 108, 129, 187, 172],
            "exang": [0, 1, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
            "slope": [0, 2, 2, 0, 1],
            "ca": [0, 3, 2, 0, 0],
            "thal": [1, 3, 7, 3, 3],
        }
    )
    return data


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values"""
    data = pd.DataFrame(
        {
            "age": [63, 67, np.nan, 37, 41],
            "sex": [1, 1, 1, np.nan, 0],
            "cp": [3, 4, 4, 3, 2],
            "trestbps": [145, 160, 120, 130, 130],
            "chol": [233, 286, np.nan, 250, 204],
            "fbs": [1, 0, 0, 0, 0],
            "restecg": [0, 2, 2, 0, 2],
            "thalach": [150, 108, 129, 187, 172],
            "exang": [0, 1, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
            "slope": [0, 2, 2, 0, 1],
            "ca": [0, 3, 2, 0, np.nan],
            "thal": [1, 3, 7, 3, 3],
        }
    )
    return data


class TestHeartDiseasePreprocessor:
    """Test suite for HeartDiseasePreprocessor"""

    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = HeartDiseasePreprocessor()
        assert preprocessor.feature_names is None
        assert preprocessor.imputer is not None
        assert preprocessor.scaler is not None
        assert len(preprocessor.expected_features) == 13

    def test_fit(self, sample_data):
        """Test fitting preprocessor"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) == 13
        assert preprocessor.imputer is not None
        assert preprocessor.scaler is not None

    def test_transform(self, sample_data):
        """Test transformation"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        transformed = preprocessor.transform(sample_data)

        # Check output type
        assert isinstance(transformed, pd.DataFrame)

        # Check shape
        assert transformed.shape == sample_data.shape

        # Check scaling (mean should be close to 0, std close to 1)
        assert abs(transformed.mean().mean()) < 0.5
        assert abs(transformed.std().mean() - 1.0) < 0.5

    def test_fit_transform(self, sample_data):
        """Test fit_transform"""
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(sample_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_data.shape

    def test_missing_value_imputation(self, sample_data_with_missing):
        """Test missing value handling"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data_with_missing)
        transformed = preprocessor.transform(sample_data_with_missing)

        # No missing values after transformation
        assert transformed.isnull().sum().sum() == 0

    def test_feature_validation(self, sample_data):
        """Test feature validation"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        # Remove a required feature
        incomplete_data = sample_data.drop("age", axis=1)

        with pytest.raises(ValueError):
            preprocessor.transform(incomplete_data)

    def test_get_feature_names(self, sample_data):
        """Test get_feature_names method"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        feature_names = preprocessor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) == 13
        assert "age" in feature_names
        assert "sex" in feature_names

    def test_save_load(self, sample_data, tmp_path):
        """Test save and load functionality"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        # Save
        save_path = tmp_path / "test_preprocessor.pkl"
        preprocessor.save(str(save_path))

        # Load
        loaded_preprocessor = HeartDiseasePreprocessor.load(str(save_path))

        # Test loaded preprocessor
        transformed_original = preprocessor.transform(sample_data)
        transformed_loaded = loaded_preprocessor.transform(sample_data)

        pd.testing.assert_frame_equal(transformed_original, transformed_loaded)

    def test_numpy_array_input(self, sample_data):
        """Test with numpy array input"""
        preprocessor = HeartDiseasePreprocessor()

        # Fit with DataFrame
        preprocessor.fit(sample_data)

        # Transform numpy array
        numpy_data = sample_data.values
        transformed = preprocessor.transform(numpy_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape[0] == numpy_data.shape[0]

    def test_reproducibility(self, sample_data):
        """Test reproducibility of transformations"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)

        # Transform twice
        transformed1 = preprocessor.transform(sample_data)
        transformed2 = preprocessor.transform(sample_data)

        # Should be identical
        pd.testing.assert_frame_equal(transformed1, transformed2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
