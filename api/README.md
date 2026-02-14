# 🩺 Retinopathy Detector API

API REST para detección de retinopatía diabética usando Deep Learning.

## 🚀 Demo en Vivo

**API URL:** `https://retinopathy-api.onrender.com`

**Documentación interactiva:** `https://retinopathy-api.onrender.com/docs`

⚠️ **Nota:** Este servicio usa Render Free Tier. La **primera request puede tardar 30-60 segundos** mientras el servicio se activa (cold start). Las siguientes requests son instantáneas.

---

## 📊 Sobre el Modelo

- **Arquitectura:** MobileNetV2 + Transfer Learning
- **Preprocesamiento:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Dataset:** RFMiD (3,200 imágenes de retina)
- **Métricas en Test Set:**
  - Recall: 85.18%
  - Accuracy: 86.72%
  - Precision: 97.73%

---

## 🔌 Endpoints

### `GET /`
Información general de la API

### `GET /health`
Verificar estado de la API y modelo

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`
Realizar predicción en imagen de retina

**Request:**
- Método: POST
- Content-Type: multipart/form-data
- Body: `file` (imagen PNG/JPG/JPEG)

**Respuesta:**
```json
{
  "prediction": "Disease Risk (Enfermo)",
  "class_id": 1,
  "confidence": 0.8518,
  "probabilities": {
    "Healthy": 0.1482,
    "Disease": 0.8518
  },
  "message": "⚠️ Se detectaron posibles signos de riesgo..."
}
```

---

## 💻 Uso con cURL
```bash
curl -X POST "https://retinopathy-api.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tu_imagen.png"
```

---

## 🐍 Uso con Python
```python
import requests

url = "https://retinopathy-api.onrender.com/predict"
files = {'file': open('retina.png', 'rb')}

response = requests.post(url, files=files)
print(response.json())
```

---

## 🔧 Ejecutar Localmente
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# Abrir docs
http://localhost:8888/docs
```

---

## 📦 Docker
```bash
# Build
docker build -t retinopathy-api .

# Run
docker run -p 8888:8888 retinopathy-api

# Abrir docs
http://localhost:8888/docs
```

---

## ⚠️ Disclaimer

Este modelo es **solo para fines educativos y demostrativos**. NO debe usarse para diagnóstico médico real. Consulte siempre con un profesional de la salud.

---

## 👨‍💻 Desarrollado por

**Carlos Saquel Depaoli**

- 🌐 Demo Web: [Retinopathy Detector en Hugging Face](https://huggingface.co/spaces/desareca/retinopathy-detector)
- 📊 Proyecto completo: [GitHub](https://github.com/desareca/retinopathy-mle)

---

## 📄 Licencia

MIT License
```

---

### **5.4 - Crear .gitignore**

Crea `~/retinopathy-api/.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/