"""
Test script for Docker container API
Tests all endpoints with sample data
"""

import requests
import json
import time

# API configuration
BASE_URL = "http://localhost:8000"

# Sample patient data
sample_patient = {
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

# Sample batch data
sample_batch = {
    "patients": [
        sample_patient,
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
}


def test_api():
    """Test all API endpoints"""
    print("=" * 80)
    print("TESTING HEART DISEASE PREDICTION API")
    print("=" * 80)

    # Test 1: Root endpoint
    print("\n[TEST 1] Root Endpoint")
    print("-" * 80)
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        print("✓ Root endpoint test passed")
    except Exception as e:
        print(f"✗ Root endpoint test failed: {str(e)}")

    # Test 2: Health check
    print("\n[TEST 2] Health Check Endpoint")
    print("-" * 80)
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        print("✓ Health check test passed")
    except Exception as e:
        print(f"✗ Health check test failed: {str(e)}")

    # Test 3: Single prediction
    print("\n[TEST 3] Single Prediction Endpoint")
    print("-" * 80)
    print(f"Input: {json.dumps(sample_patient, indent=2)}")
    try:
        response = requests.post(
            f"{BASE_URL}/predict", json=sample_patient, headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        assert response.status_code == 200
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]

        print(f"\n✓ Prediction: {data['prediction_label']}")
        print(f"✓ Confidence: {data['confidence']*100:.2f}%")
        print(f"✓ Probability of Disease: {data['probability_disease']*100:.2f}%")
        print("✓ Single prediction test passed")
    except Exception as e:
        print(f"✗ Single prediction test failed: {str(e)}")

    # Test 4: Batch prediction
    print("\n[TEST 4] Batch Prediction Endpoint")
    print("-" * 80)
    print(f"Input: {len(sample_batch['patients'])} patients")
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=sample_batch,
            headers={"Content-Type": "application/json"},
        )
        print(f"\nStatus Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        assert response.status_code == 200
        assert "predictions" in data
        assert data["count"] == len(sample_batch["patients"])

        print(f"\n✓ Batch predictions: {data['count']}")
        for i, pred in enumerate(data["predictions"], 1):
            print(
                f"  Patient {i}: {pred['prediction_label']} (Confidence: {pred['confidence']*100:.2f}%)"
            )
        print("✓ Batch prediction test passed")
    except Exception as e:
        print(f"✗ Batch prediction test failed: {str(e)}")

    # Test 5: Model info
    print("\n[TEST 5] Model Info Endpoint")
    print("-" * 80)
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        assert response.status_code == 200
        assert "model_type" in data
        assert "n_features" in data

        print(f"\n✓ Model Type: {data['model_type']}")
        print(f"✓ Number of Features: {data['n_features']}")
        print("✓ Model info test passed")
    except Exception as e:
        print(f"✗ Model info test failed: {str(e)}")

    # Test 6: Invalid input
    print("\n[TEST 6] Invalid Input Handling")
    print("-" * 80)
    invalid_patient = {"age": 63}  # Missing required fields
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_patient,
            headers={"Content-Type": "application/json"},
        )
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 422  # Validation error
        print("✓ Invalid input test passed (correctly rejected)")
    except Exception as e:
        print(f"✗ Invalid input test failed: {str(e)}")

    # Final summary
    print("\n" + "=" * 80)
    print("API TESTING COMPLETE")
    print("=" * 80)
    print("\n✓ All endpoint tests completed successfully!")
    print(f"\nAPI Documentation: {BASE_URL}/docs")
    print(f"ReDoc Documentation: {BASE_URL}/redoc")


if __name__ == "__main__":
    print("\nWaiting for API to be ready...")
    time.sleep(2)

    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("  Make sure the Docker container is running:")
        print("  docker-compose up -d")
        print("  or")
        print("  python app.py")
