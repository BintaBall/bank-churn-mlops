# app/main.py - VERSION AVEC CACHE
from functools import lru_cache
import hashlib
import json
from typing import Dict, Any
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
import joblib
import numpy as np
import logging
import os
import traceback

# Import conditionnel pour Azure
APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

# ============================================================
# LOGGING & APPLICATION INSIGHTS
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bank-churn-api")

if APPINSIGHTS_CONN:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
        logger.addHandler(handler)
        logger.info("Azure Application Insights connecté", extra={
            "custom_dimensions": {
                "event_type": "startup",
                "status": "application_insights_connected"
            }
        })
    except ImportError:
        logger.warning("opencensus non installé - logs locaux uniquement", extra={
            "custom_dimensions": {
                "event_type": "startup",
                "status": "application_insights_not_configured"
            }
        })
else:
    logger.info("Application Insights non configuré - logs locaux", extra={
        "custom_dimensions": {
            "event_type": "startup",
            "status": "application_insights_not_configured"
        }
    })

# ============================================================
# MODÈLES PYDANTIC
# ============================================================

try:
    from app.models import CustomerFeatures, PredictionResponse, HealthResponse
    from app.drift_detect import detect_drift
except ImportError as e:
    logger.error(f"Erreur import modèles: {e}")
    from pydantic import BaseModel
    
    class CustomerFeatures(BaseModel):
        CreditScore: int; Age: int; Tenure: int; Balance: float
        NumOfProducts: int; HasCrCard: int; IsActiveMember: int
        EstimatedSalary: float; Geography_Germany: int; Geography_Spain: int
        
    class PredictionResponse(BaseModel):
        churn_probability: float; prediction: int; risk_level: str
        
    class HealthResponse(BaseModel):
        status: str; model_loaded: bool
    
    def detect_drift(**kwargs):
        logger.warning("Module drift_detect non disponible")
        return {"error": "drift_detect_module_not_available"}

# ============================================================
# FONCTIONS DE CACHE
# ============================================================

def hash_features(features_dict: Dict[str, Any]) -> str:
    """Crée un hash MD5 unique pour les features"""
    features_str = json.dumps(features_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(features_str.encode()).hexdigest()

class ModelPredictor:
    """Classe wrapper pour le modèle avec cache"""
    
    def __init__(self, model):
        self.model = model
        self._cache_stats = {"hits": 0, "misses": 0, "size": 0}
        
    @lru_cache(maxsize=2000)  # Cache 2000 prédictions
    def _cached_predict(self, features_hash: str, *feature_values):
        """Méthode interne avec cache LRU"""
        input_array = np.array(feature_values, dtype=np.float32).reshape(1, -1)
        
        try:
            proba = float(self.model.predict_proba(input_array)[0, 1])
            prediction = int(proba > 0.5)
            
            # Log pour monitoring cache
            logger.debug(f"Cache MISS - Calcul prédiction: {features_hash[:8]}")
            
            return proba, prediction
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            raise
    
    def predict_with_cache(self, features: CustomerFeatures, features_hash: str = None):
        """Prédiction avec cache optimisé"""
        if features_hash is None:
            features_hash = hash_features(features.dict())
        
        # Extraire les features dans l'ordre fixe
        feature_values = (
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
        )
        
        # Vérifier d'abord si dans le cache
        cache_info = self._cached_predict.cache_info()
        cache_hit_before = cache_info.hits
        
        # Appel au cache
        proba, prediction = self._cached_predict(features_hash, *feature_values)
        
        # Mise à jour des stats
        cache_info = self._cached_predict.cache_info()
        if cache_info.hits > cache_hit_before:
            self._cache_stats["hits"] += 1
            cache_status = "HIT"
        else:
            self._cache_stats["misses"] += 1
            cache_status = "MISS"
        
        self._cache_stats["size"] = cache_info.currsize
        
        # Déterminer le niveau de risque
        if proba < 0.3:
            risk = "Low"
        elif proba < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        return {
            "churn_probability": round(proba, 4),
            "prediction": prediction,
            "risk_level": risk,
            "cache_hit": cache_status,
            "cache_hash": features_hash[:12]
        }
    
    def get_cache_stats(self):
        """Retourne les statistiques du cache"""
        cache_info = self._cached_predict.cache_info()
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_ratio": (
                self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
                if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0
            ),
            "cache_size": cache_info.currsize,
            "cache_maxsize": cache_info.maxsize,
            "hits_total": cache_info.hits,
            "misses_total": cache_info.misses
        }
    
    def clear_cache(self):
        """Vide le cache"""
        self._cached_predict.cache_clear()
        self._cache_stats = {"hits": 0, "misses": 0, "size": 0}
        logger.info("Cache vidé")

# ============================================================
# LIFECYCLE ET INITIALISATION
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = None
model_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du lifecycle de l'application (moderne)"""
    global model, model_predictor
    
    # Startup
    logger.info("Démarrage de l'application...")
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Fichier modèle introuvable: {MODEL_PATH}")
        else:
            model = joblib.load(MODEL_PATH)
            model_predictor = ModelPredictor(model)
            logger.info(f"Modèle chargé depuis {MODEL_PATH}", extra={
                "custom_dimensions": {
                    "event_type": "model_load",
                    "model_path": MODEL_PATH,
                    "status": "success"
                }
            })
            
            # Validation
            try:
                test_features = CustomerFeatures(
                    CreditScore=650, Age=42, Tenure=5, Balance=12500.50,
                    NumOfProducts=2, HasCrCard=1, IsActiveMember=1,
                    EstimatedSalary=45000.00, Geography_Germany=0, Geography_Spain=1
                )
                test_result = model_predictor.predict_with_cache(test_features)
                logger.info(f"Modèle validé: {test_result}")
            except Exception as e:
                logger.error(f"Erreur validation modèle: {e}")
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}", exc_info=True)
        model = None
        model_predictor = None
    
    yield
    
    # Shutdown
    logger.info("Arrêt de l'application...")
    if model_predictor:
        stats = model_predictor.get_cache_stats()
        logger.info(f"Statistiques cache finales: {stats}")

# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prédiction du churn client avec cache LRU",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Utilise le lifespan manager moderne
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENV") == "development" else [],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# ENDPOINTS GÉNÉRAUX
# ============================================================

@app.get("/")
async def root():
    """Endpoint racine avec infos cache"""
    cache_stats = model_predictor.get_cache_stats() if model_predictor else {}
    
    return {
        "message": "Bank Churn Prediction API v2",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "cache_enabled": model_predictor is not None,
        "cache_stats": cache_stats if model_predictor else None,
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "cache_stats": "/cache/stats",
            "drift_check": "/drift/check"
        }
    }

@app.get("/health")
async def health():
    """Health check avec infos cache"""
    if model is None or model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import psutil
    import time as ttime
    
    cache_stats = model_predictor.get_cache_stats()
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "cache_enabled": True,
        "cache_hit_ratio": f"{cache_stats['hit_ratio']:.2%}",
        "cache_size": cache_stats["cache_size"],
        "memory_usage": f"{memory.percent}%",
        "uptime": ttime.time() - app_start_time,
        "timestamp": os.getenv("DEPLOY_TIMESTAMP", ttime.ctime())
    }

# ============================================================
# ENDPOINTS DE PRÉDICTION AVEC CACHE
# ============================================================

@app.post("/predict")
async def predict(features: CustomerFeatures):
    """Prédiction avec cache LRU"""
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    start_time = time.time()
    
    try:
        # Prédiction avec cache
        result = model_predictor.predict_with_cache(features)
        
        # Calcul du temps d'exécution
        execution_time = (time.time() - start_time) * 1000  # en ms
        
        # Log avec infos cache
        logger.info(f"Prediction - Cache: {result['cache_hit']}, Time: {execution_time:.2f}ms", extra={
            "custom_dimensions": {
                "event_type": "prediction",
                "endpoint": "/predict",
                "probability": result["churn_probability"],
                "prediction": result["prediction"],
                "risk_level": result["risk_level"],
                "cache_hit": result["cache_hit"],
                "execution_time_ms": execution_time,
                "cache_hash": result["cache_hash"]
            }
        })
        
        # Retourne sans les infos cache internes
        return {
            "churn_probability": result["churn_probability"],
            "prediction": result["prediction"],
            "risk_level": result["risk_level"],
            "cache_info": {
                "hit": result["cache_hit"] == "HIT",
                "response_time_ms": round(execution_time, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}", extra={
            "custom_dimensions": {
                "event_type": "prediction_error",
                "error": str(e)
            }
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(features_list: List[CustomerFeatures]):
    """Prédiction par lot avec optimisation cache"""
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    start_time = time.time()
    
    try:
        predictions = []
        cache_hits = 0
        cache_misses = 0
        
        for features in features_list:
            result = model_predictor.predict_with_cache(features)
            predictions.append({
                "churn_probability": result["churn_probability"],
                "prediction": result["prediction"],
                "cache_hit": result["cache_hit"] == "HIT"
            })
            
            if result["cache_hit"] == "HIT":
                cache_hits += 1
            else:
                cache_misses += 1
        
        execution_time = (time.time() - start_time) * 1000
        total_items = len(predictions)
        
        logger.info(f"Batch prediction - Items: {total_items}, Cache hits: {cache_hits}", extra={
            "custom_dimensions": {
                "event_type": "batch_prediction",
                "count": total_items,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_ratio": f"{(cache_hits/total_items)*100:.1f}%" if total_items > 0 else "0%",
                "execution_time_ms": execution_time
            }
        })
        
        return {
            "predictions": predictions,
            "count": total_items,
            "cache_stats": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_ratio": f"{(cache_hits/total_items)*100:.1f}%" if total_items > 0 else "0%",
                "total_time_ms": round(execution_time, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur batch prediction: {e}", extra={
            "custom_dimensions": {
                "event_type": "batch_prediction_error",
                "error": str(e)
            }
        })
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ENDPOINTS DE GESTION DU CACHE
# ============================================================

@app.get("/cache/stats")
async def get_cache_stats():
    """Retourne les statistiques du cache"""
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = model_predictor.get_cache_stats()
    
    return {
        "status": "success",
        "cache_enabled": True,
        "stats": stats,
        "timestamp": time.time()
    }

@app.post("/cache/clear")
async def clear_cache():
    """Vide le cache"""
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_predictor.clear_cache()
    
    logger.info("Cache vidé manuellement", extra={
        "custom_dimensions": {
            "event_type": "cache_clear",
            "triggered_by": "api_endpoint"
        }
    })
    
    return {
        "status": "success",
        "message": "Cache cleared",
        "timestamp": time.time()
    }

# ============================================================
# DRIFT DETECTION (inchangé)
# ============================================================

def log_drift_to_insights(drift_results: dict):
    """Envoie les résultats de drift à Application Insights"""
    if not drift_results or "error" in drift_results:
        return
    
    total = len(drift_results)
    drifted = sum(1 for r in drift_results.values() if r.get("drift_detected", False))
    percentage = round((drifted / total) * 100, 2) if total else 0
    risk = "LOW" if percentage < 20 else "MEDIUM" if percentage < 50 else "HIGH"
    
    logger.warning(
        f"Drift detection: {drifted}/{total} features drifted ({percentage}%)",
        extra={
            "custom_dimensions": {
                "event_type": "drift_detection",
                "drift_percentage": percentage,
                "risk_level": risk,
                "features_analyzed": total,
                "features_drifted": drifted
            }
        }
    )

@app.post("/drift/check")
async def check_drift(threshold: float = 0.05):
    """Endpoint pour vérifier le drift de données"""
    try:
        results = detect_drift(
            reference_file="data/bank_churn.csv",
            production_file="data/production_data.csv",
            threshold=threshold
        )
        
        log_drift_to_insights(results)
        
        if "error" in results:
            return {
                "status": "error",
                "message": results["error"],
                "features_analyzed": 0,
                "features_drifted": 0
            }
        
        drifted = sum(1 for r in results.values() if r.get("drift_detected", False))
        
        return {
            "status": "success",
            "features_analyzed": len(results),
            "features_drifted": drifted,
            "drift_percentage": round((drifted / len(results)) * 100, 2) if results else 0,
            "risk_level": "LOW" if drifted < 0.2 * len(results) else "MEDIUM" if drifted < 0.5 * len(results) else "HIGH"
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Erreur drift check: {e}", extra={
            "custom_dimensions": {
                "event_type": "drift_error",
                "traceback": tb
            }
        })
        raise HTTPException(status_code=500, detail="Drift check failed")

@app.post("/drift/alert")
async def manual_drift_alert(
    message: str = "Manual drift alert triggered",
    severity: str = "warning"
):
    """Endpoint pour déclencher manuellement une alerte de drift"""
    logger.warning(f"Alerte drift manuelle: {message}", extra={
        "custom_dimensions": {
            "event_type": "manual_drift_alert",
            "alert_message": message,
            "severity": severity,
            "triggered_by": "api_endpoint"
        }
    })
    
    return {"status": "alert_sent", "message": message, "severity": severity}

# ============================================================
# VARIABLE GLOBALE POUR STARTUP TIME
# ============================================================

app_start_time = time.time()

# ============================================================
# MIDDLEWARE
# ============================================================

@app.middleware("http")
async def add_security_headers(request, call_next):
    """Ajoute des headers de sécurité basiques"""
    response = await call_next(request)
    
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