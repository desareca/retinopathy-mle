"""
FastAPI REST API for Retinopathy Detector
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Página principal con información de la API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Retinopathy Detector API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 10px auto;
                padding: 10px;
                background: white;
                color: white;
            }
            .container {
                background: linear-gradient(135deg, #667eeadd 0%, #4b52a2dd 100%);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { margin-top: 0; }
            .endpoint {
                background: rgba(0, 0, 0, 0.2);
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            }
            .endpoint code {
                background: rgba(255, 255, 255, 0.2);
                padding: 2px 8px;
                border-radius: 4px;
            }
            .warning {
                background: rgba(255, 200, 0, 0.3);
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #ffc107;
            }
            a {
                color: #c1e2ff;
                text-decoration: none;
                font-weight: bold;
            }
            a:hover { text-decoration: underline; }
            .badge {
                display: inline-block;
                padding: 5px 10px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                font-size: 0.9em;
                margin: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 Retinopathy Detector API</h1>
            <p>API REST para detección de retinopatía diabética usando Deep Learning</p>
            
            <div class="badge">Version 1.0.0</div>
            <div class="badge">FastAPI</div>
            <div class="badge">TensorFlow 2.15</div>
            
            <div class="warning">
                <strong>⚠️ Free Tier Notice:</strong><br>
                La primera request después de 15 minutos de inactividad tardará 
                <strong>30-60 segundos</strong> mientras el servicio se reactiva (cold start).
            </div>
            
            <h2>📚 Endpoints</h2>
            
            <div class="endpoint">
                <strong>GET /health</strong><br>
                Verificar estado de la API y modelo
            </div>
            
            <div class="endpoint">
                <strong>POST /predict</strong><br>
                Realizar predicción en imagen de retina
            </div>
            
            <div class="endpoint">
                <strong>GET /docs</strong><br>
                Documentación interactiva (Swagger UI)
            </div>
            
            <h2>🚀 Pruébala Ahora</h2>
            <p>
                <a href="https://retinopathy-api.onrender.com/docs" target="_blank">→ Abrir Documentación Interactiva</a>
            </p>
            
            <h2>📊 Métricas del Modelo</h2>
            <ul>
                <li><strong>Recall:</strong> 85.18%</li>
                <li><strong>Accuracy:</strong> 86.72%</li>
                <li><strong>Precision:</strong> 97.73%</li>
            </ul>
            
            <h2>🔗 Links</h2>
            <p>
                <a href="https://huggingface.co/spaces/desareca/retinopathy-detector">Demo Visual</a> |
                <a href="https://github.com/desareca/retinopathy-api">GitHub API</a> |
                <a href="https://github.com/desareca/retinopathy-mle">GitHub Proyecto</a>
            </p>
            
            <p style="margin-top: 30px; font-size: 0.9em; opacity: 0.9;">
                Desarrollado por Carlos Saquel Depaoli
            </p>
        </div>
    </body>
    </html>
    """
    return html_content


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