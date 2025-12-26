"""
Unit Tests for Prediction Interface
Tests the HeartDiseasePredictor class
"""

import pytest
import pandas as pd

from predict import HeartDiseasePredictor


@pytest.fixture
def sample_patient():
    """Sample patient data"""
    return {
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


@pytest.fixture
def sample_patients_df():
    """Sample patients as DataFrame"""
    return pd.DataFrame(
        [
            {
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
            },
            {
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
            },
        ]
    )


class TestHeartDiseasePredictor:
    """Test suite for HeartDiseasePredictor"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup predictor before each test"""
        self.predictor = HeartDiseasePredictor()

    def test_initialization(self):
        """Test predictor initialization"""
        assert self.predictor.model is not None
        assert self.predictor.preprocessor is not None
        assert len(self.predictor.required_features) == 13

    def test_validate_input_dict(self, sample_patient):
        """Test input validation with dictionary"""
        validated = self.predictor.validate_input(sample_patient)

        assert isinstance(validated, pd.DataFrame)
        assert len(validated) == 1
        assert all(col in validated.columns for col in self.predictor.required_features)

    def test_validate_input_dataframe(self, sample_patients_df):
        """Test input validation with DataFrame"""
        validated = self.predictor.validate_input(sample_patients_df)

        assert isinstance(validated, pd.DataFrame)
        assert len(validated) == 2
        assert all(col in validated.columns for col in self.predictor.required_features)

    def test_validate_input_missing_features(self):
        """Test validation with missing features"""
        incomplete_patient = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            # Missing other features
        }

        with pytest.raises(ValueError, match="Missing required features"):
            self.predictor.validate_input(incomplete_patient)

    def test_predict_single(self, sample_patient):
        """Test single prediction"""
        result = self.predictor.predict(sample_patient)

        # Check result structure
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "prediction_label" in result
        assert "probability_no_disease" in result
        assert "probability_disease" in result
        assert "confidence" in result

        # Check value types
        assert isinstance(result["prediction"], int)
        assert result["prediction"] in [0, 1]
        assert isinstance(result["prediction_label"], str)
        assert result["prediction_label"] in ["Disease", "No Disease"]

        # Check probabilities
        assert 0 <= result["probability_no_disease"] <= 1
        assert 0 <= result["probability_disease"] <= 1
        assert abs(result["probability_no_disease"] + result["probability_disease"] - 1.0) < 0.01
        assert 0.5 <= result["confidence"] <= 1.0

    def test_predict_without_proba(self, sample_patient):
        """Test prediction without probabilities"""
        result = self.predictor.predict(sample_patient, return_proba=False)

        assert "prediction" in result
        assert "prediction_label" in result
        assert "probability_no_disease" not in result
        assert "probability_disease" not in result

    def test_predict_batch(self, sample_patients_df):
        """Test batch prediction"""
        results = self.predictor.predict_batch(sample_patients_df)

        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert "prediction" in result
            assert "prediction_label" in result
            assert result["prediction"] in [0, 1]

    def test_predict_batch_list(self):
        """Test batch prediction with list of dicts"""
        patients = [
            {
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
            },
            {
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
            },
        ]

        results = self.predictor.predict_batch(patients)

        assert len(results) == 2
        assert all("prediction" in r for r in results)

    def test_prediction_consistency(self, sample_patient):
        """Test prediction consistency (same input = same output)"""
        result1 = self.predictor.predict(sample_patient)
        result2 = self.predictor.predict(sample_patient)

        assert result1["prediction"] == result2["prediction"]
        assert abs(result1["confidence"] - result2["confidence"]) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
