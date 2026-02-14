"""
FastAPI REST API for Retinopathy Detector
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Retinopathy Detector API",
    description="API REST para detección de retinopatía diabética usando Deep Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo (solo una vez al iniciar)
MODEL_PATH = "model.h5"
model = None

@app.on_event("startup")
async def load_model():
    """Cargar modelo al iniciar la aplicación"""
    global model
    try:
        logger.info(f"Cargando modelo desde {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✓ Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        raise


# Modelos Pydantic para request/response
class PredictionResponse(BaseModel):
    """Modelo de respuesta de predicción"""
    prediction: str
    class_id: int
    confidence: float
    probabilities: dict
    message: str


class HealthResponse(BaseModel):
    """Modelo de respuesta de health check"""
    status: str
    model_loaded: bool
    version: str


def preprocess_clahe(image_array: np.ndarray) -> np.ndarray:
    """
    Aplicar preprocesamiento CLAHE a la imagen
    
    Args:
        image_array: Imagen numpy array (H, W, 3)
        
    Returns:
        Imagen preprocesada (224, 224, 3) normalizada [0, 1]
    """
    # Resize a 224x224
    img_resized = cv2.resize(image_array, (224, 224))
    
    # Convertir a LAB
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    
    # Aplicar CLAHE solo al canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convertir de vuelta a RGB
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalizar a [0, 1]
    img_normalized = img_clahe.astype(np.float32) / 255.0
    
    return img_normalized


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "message": "Retinopathy Detector API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        },
        "note": "⚠️ Free tier: Primera request puede tardar 30-60s (cold start)"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Verificar estado de la API y modelo
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Realizar predicción en una imagen de retina
    
    - **file**: Imagen de retina (PNG, JPG, JPEG)
    
    Retorna:
    - **prediction**: "Healthy (Sano)" o "Disease Risk (Enfermo)"
    - **class_id**: 0 (sano) o 1 (enfermo)
    - **confidence**: Confianza de la predicción (0-1)
    - **probabilities**: Probabilidades para ambas clases
    """
    
    # Verificar que el modelo esté cargado
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Intenta nuevamente en unos segundos."
        )
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo inválido: {file.content_type}. Usa PNG, JPG o JPEG."
        )
    
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a numpy array
        image_array = np.array(image)
        
        # Preprocesar
        processed_image = preprocess_clahe(image_array)
        
        # Agregar batch dimension
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Predicción
        predictions = model.predict(image_batch, verbose=0)
        
        # Extraer resultados
        class_id = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_id])
        
        # Nombres de clases
        class_names = {
            0: "Healthy (Sano)",
            1: "Disease Risk (Enfermo)"
        }
        
        # Mensaje según resultado
        if class_id == 0:
            message = "✅ La imagen sugiere una retina saludable. Mantener controles regulares."
        else:
            message = "⚠️ Se detectaron posibles signos de riesgo. Consulte con un oftalmólogo profesional."
        
        return PredictionResponse(
            prediction=class_names[class_id],
            class_id=class_id,
            confidence=confidence,
            probabilities={
                "Healthy": float(predictions[0][0]),
                "Disease": float(predictions[0][1])
            },
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador personalizado de excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)