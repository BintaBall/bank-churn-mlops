# app/main.py - VERSION OPTIMISÉE POUR PRODUCTION
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import numpy as np
import logging
import os
import json
import traceback
from pathlib import Path

# Import conditionnel pour Azure (inutile en dev local)
APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

# ============================================================
# LOGGING OPTIMISÉ
# ============================================================

# Configuration minimale du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bank-churn-api")

# Application Insights uniquement si configuré
if APPINSIGHTS_CONN:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
        logger.addHandler(handler)
        logger.info("Azure Application Insights connecté")
    except ImportError:
        logger.warning("opencensus non installé - logs locaux uniquement")
else:
    logger.info("Application Insights non configuré - logs locaux")

# ============================================================
# MODÈLES PYDANTIC
# ============================================================

try:
    from app.models import CustomerFeatures, PredictionResponse, HealthResponse
except ImportError as e:
    logger.error(f"Erreur import modèles: {e}")
    # Fallback simple si besoin
    from pydantic import BaseModel
    class CustomerFeatures(BaseModel):
        CreditScore: int; Age: int; Tenure: int; Balance: float
        NumOfProducts: int; HasCrCard: int; IsActiveMember: int
        EstimatedSalary: float; Geography_Germany: int; Geography_Spain: int
        
    class PredictionResponse(BaseModel):
        churn_probability: float; prediction: int; risk_level: str
        
    class HealthResponse(BaseModel):
        status: str; model_loaded: bool

# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prédiction du churn client",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENV") == "development" else [],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# CONFIGURATION MODÈLE
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Fichier modèle introuvable: {MODEL_PATH}")
            model = None
            return
            
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        
        # Validation rapide du modèle
        if hasattr(model, "predict"):
            test_input = np.zeros((1, 10))  # 10 features
            try:
                _ = model.predict(test_input)
                logger.info("Modèle validé avec succès")
            except Exception as e:
                logger.error(f"Erreur validation modèle: {e}")
        else:
            logger.error("Modèle invalide - méthode predict manquante")
            
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}", exc_info=True)
        model = None

# ============================================================
# ENDPOINTS ESSENTIELS
# ============================================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "service": "Bank Churn Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check simplifié"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - model not loaded"
        )
    
    # Vérification de mémoire
    import psutil
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "memory_usage": f"{memory.percent}%",
        "timestamp": os.getenv("DEPLOY_TIMESTAMP", "unknown")
    }

@app.post("/predict")
async def predict(features: CustomerFeatures):
    """Prédiction unique optimisée"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # Conversion directe sans tableau intermédiaire complexe
        input_features = [
            features.CreditScore,
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary,
            features.Geography_Germany,
            features.Geography_Spain
        ]
        
        # Utilisation de reshape au lieu de array nesting
        input_array = np.array(input_features, dtype=np.float32).reshape(1, -1)
        
        # Prédiction
        proba = float(model.predict_proba(input_array)[0, 1])
        prediction = int(proba > 0.5)
        
        # Log minimal
        logger.info(f"Prediction - proba: {proba:.4f}, result: {prediction}")
        
        return {
            "churn_probability": round(proba, 4),
            "prediction": prediction,
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/predict/batch")
async def predict_batch(features_list: List[CustomerFeatures]):
    """Prédiction par lot optimisée"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        predictions = []
        
        # Pré-allocation pour performance
        for features in features_list:
            input_features = [
                features.CreditScore,
                features.Age,
                features.Tenure,
                features.Balance,
                features.NumOfProducts,
                features.HasCrCard,
                features.IsActiveMember,
                features.EstimatedSalary,
                features.Geography_Germany,
                features.Geography_Spain
            ]
            
            input_array = np.array(input_features, dtype=np.float32).reshape(1, -1)
            proba = float(model.predict_proba(input_array)[0, 1])
            
            predictions.append({
                "churn_probability": round(proba, 4),
                "prediction": int(proba > 0.5)
            })
        
        logger.info(f"Batch prediction - count: {len(predictions)}")
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "batch_id": os.urandom(4).hex()  # ID unique pour tracing
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal batch prediction error")

# ============================================================
# FONCTIONS DÉPLACÉES POUR DÉPLOIEMENT FUTUR
# ============================================================

def log_drift_to_insights(drift_results: dict):
    """Placeholder pour la détection de drift (à implémenter avec Azure)"""
    logger.info(f"Drift detection placeholder - {len(drift_results)} features")
    # À implémenter avec Azure Application Insights

@app.post("/drift/check")
async def check_drift(threshold: float = 0.05):
    """Endpoint drift (pour futur déploiement Azure)"""
    return {
        "status": "drift_detection_disabled",
        "message": "Enable with AZURE_DRIFT_ENABLED=true",
        "features_analyzed": 0,
        "features_drifted": 0
    }

# ============================================================
# MIDDLEWARE DE SÉCURITÉ BASIQUE
# ============================================================

@app.middleware("http")
async def add_security_headers(request, call_next):
    """Ajoute des headers de sécurité basiques"""
    response = await call_next(request)
    
    # Headers pour production
    if os.getenv("ENV") == "production":
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
    
    return response

# ============================================================
# POINT D'ENTRÉE ALTERNATIF
# ============================================================

if __name__ == "__main__":
    """Pour exécution directe (développement uniquement)"""
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV") == "development"
    )