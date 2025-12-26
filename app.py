"""
FastAPI Application for Heart Disease Prediction
Containerized model serving with REST API, Logging, and Monitoring
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from preprocessing_pipeline import HeartDiseasePreprocessor
import uvicorn
from datetime import datetime
import logging
import sys
import time
import types

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from starlette.responses import Response

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics - avoid duplicate registration when module is reloaded
def _get_collector(name):
    return REGISTRY._names_to_collectors.get(name)

if _get_collector('api_requests_total') is None:
    REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
else:
    REQUEST_COUNT = _get_collector('api_requests_total')

if _get_collector('api_request_latency_seconds') is None:
    REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency in seconds', ['method', 'endpoint'])
else:
    REQUEST_LATENCY = _get_collector('api_request_latency_seconds')

if _get_collector('predictions_total') is None:
    PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['prediction_label'])
else:
    PREDICTION_COUNT = _get_collector('predictions_total')

if _get_collector('prediction_confidence') is None:
    PREDICTION_CONFIDENCE = Histogram('prediction_confidence', 'Prediction confidence scores', buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
else:
    PREDICTION_CONFIDENCE = _get_collector('prediction_confidence')

if _get_collector('model_loaded') is None:
    MODEL_LOAD_STATUS = Gauge('model_loaded', 'Model load status (1=loaded, 0=not loaded)')
else:
    MODEL_LOAD_STATUS = _get_collector('model_loaded')

if _get_collector('active_requests') is None:
    ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
else:
    ACTIVE_REQUESTS = _get_collector('active_requests')

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API for predicting heart disease risk with monitoring",
    version="1.0.0",
    contact={
        "name": "MLOps Team",
        "email": "mlops@example.com"
    }
)

# Load model and preprocessor at startup
MODEL_PATH = "packaged_models/pickle/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None


# Middleware for logging and metrics
@app.middleware("http")
async def log_and_monitor_requests(request: Request, call_next):
    """Middleware to log all requests and track metrics"""
    start_time = time.time()
    
    # Increment active requests
    ACTIVE_REQUESTS.inc()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(latency)
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status={response.status_code} Latency={latency:.3f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, preprocessor
    
    try:
        logger.info("Loading model and preprocessor...")
        model = joblib.load(MODEL_PATH)
        # Compatibility: some preprocessor pickles were created when
        # the class was defined under __main__ (script run directly).
        # When running under uvicorn the __main__ module is different,
        # so make the class available on __main__ to allow unpickling.
        if '__main__' not in sys.modules:
            sys.modules['__main__'] = types.ModuleType('__main__')
        setattr(sys.modules['__main__'], 'HeartDiseasePreprocessor', HeartDiseasePreprocessor)

        preprocessor = HeartDiseasePreprocessor.load(PREPROCESSOR_PATH)
        MODEL_LOAD_STATUS.set(1)
        logger.info(f"✓ Model loaded: {type(model).__name__}")
        logger.info(f"✓ Preprocessor loaded")
    except Exception as e:
        logger.error(f"✗ Error loading model: {str(e)}", exc_info=True)
        MODEL_LOAD_STATUS.set(0)
        raise


# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Patient data input schema"""
    age: int = Field(..., ge=29, le=77, description="Age in years (29-77)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=94, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=126, le=564, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=71, le=202, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0/1)")
    oldpeak: float = Field(..., ge=0, le=6.2, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia (0-7)")
    
    class Config:
        schema_extra = {
            "example": {
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
                "thal": 1
            }
        }


class BatchPatientData(BaseModel):
    """Batch prediction input schema"""
    patients: List[PatientData] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    prediction: int = Field(..., description="Predicted class (0=No Disease, 1=Disease)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_no_disease: float = Field(..., description="Probability of no disease")
    probability_disease: float = Field(..., description="Probability of disease")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    predictions: List[PredictionResponse]
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    timestamp: str


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None and preprocessor is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Predict heart disease for a single patient
    
    Parameters:
    - patient: Patient clinical data
    
    Returns:
    - Prediction with confidence scores
    """
    if model is None or preprocessor is None:
        logger.error("Prediction failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Prediction request received: age={patient.age}, sex={patient.sex}")
        
        # Convert to DataFrame
        patient_dict = patient.dict()
        patient_df = pd.DataFrame([patient_dict])
        
        # Preprocess
        patient_processed = preprocessor.transform(patient_df)
        
        # Predict
        prediction = int(model.predict(patient_processed)[0])
        probabilities = model.predict_proba(patient_processed)[0]
        
        prediction_label = "Disease" if prediction == 1 else "No Disease"
        confidence = float(max(probabilities))
        
        # Record metrics
        PREDICTION_COUNT.labels(prediction_label=prediction_label).inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        
        # Prepare response
        response = {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "probability_no_disease": float(probabilities[0]),
            "probability_disease": float(probabilities[1]),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Prediction successful: {prediction_label} "
            f"(confidence={confidence:.2%})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchPatientData):
    """
    Predict heart disease for multiple patients
    
    Parameters:
    - batch: List of patient clinical data
    
    Returns:
    - List of predictions with confidence scores
    """
    if model is None or preprocessor is None:
        logger.error("Batch prediction failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Batch prediction request: {len(batch.patients)} patients")
        
        # Convert to DataFrame
        patients_data = [p.dict() for p in batch.patients]
        patients_df = pd.DataFrame(patients_data)
        
        # Preprocess
        patients_processed = preprocessor.transform(patients_df)
        
        # Predict
        predictions = model.predict(patients_processed)
        probabilities = model.predict_proba(patients_processed)
        
        # Prepare responses
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            prediction_label = "Disease" if pred == 1 else "No Disease"
            confidence = float(max(proba))
            
            # Record metrics
            PREDICTION_COUNT.labels(prediction_label=prediction_label).inc()
            PREDICTION_CONFIDENCE.observe(confidence)
            
            result = {
                "prediction": int(pred),
                "prediction_label": prediction_label,
                "probability_no_disease": float(proba[0]),
                "probability_disease": float(proba[1]),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
        
        logger.info(f"Batch prediction successful: {len(results)} predictions made")
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
        "features": preprocessor.get_feature_names() if preprocessor else None,
        "n_features": len(preprocessor.get_feature_names()) if preprocessor else 0
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
