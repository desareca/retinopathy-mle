# 🩺 Retinopathy Detection - ML End-to-End Project

**Clasificación binaria de imágenes de retina para detección de retinopatía diabética usando Deep Learning.**

[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/desareca/retinopathy-detector)
[![API](https://img.shields.io/badge/API-Render-blue)](https://retinopathy-api.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📊 Resultados Finales

| Métrica | Valor |
|---------|-------|
| **Recall (Clase Enfermo)** | **85.18%** |
| **Accuracy** | **86.72%** |
| **Precision** | **97.73%** |

**Modelo:** MobileNetV2 + Fine-tuning (CLAHE preprocessing, unfreeze_from=120)  
**Dataset:** RFMiD - 3,200 imágenes de fondo de ojo

---

## 🎯 Demos en Vivo

### 🖼️ Interfaz Web Interactiva
Prueba el modelo visualmente subiendo imágenes:

**→ [Demo en Hugging Face Spaces](https://huggingface.co/spaces/desareca/retinopathy-detector)**

### 🔌 API REST
Integra el modelo en tus aplicaciones:

**→ [API Documentation (Swagger)](https://retinopathy-api.onrender.com/docs)**

⚠️ **Nota:** API en Render Free Tier - primera request puede tardar 30-60s (cold start)

---

## 🏗️ Arquitectura del Proyecto
```
┌─────────────────────────────────────────────────────────────┐
│                     DATASET RFMiD                           │
│              3,200 imágenes de retina                       │
│         1,920 train | 640 val | 640 test                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Preprocesamiento      │
         │  • CLAHE               │
         │  • Resize 224x224      │
         │  • Normalización       │
         └────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  MobileNetV2 (ImageNet)     │
    │  • 154 capas totales        │
    │  • Transfer Learning        │
    │  • Fine-tuning desde capa   │
    │    120 (últimas 34 capas)   │
    └────────┬────────────────────┘
             │
             ▼
   ┌──────────────────────────┐
   │  Custom Classification   │
   │  Head                    │
   │  • GAP                   │
   │  • Dropout(0.3)          │
   │  • Dense(128, relu)      │
   │  • Dropout(0.3)          │
   │  • Dense(2, softmax)     │
   └────────┬─────────────────┘
            │
            ▼
  ┌───────────────────────────┐
  │  Deployment               │
  │  • Gradio UI (HF Spaces)  │
  │  • FastAPI (Render)       │
  └───────────────────────────┘
```

---

## 🔬 Pipeline de Desarrollo

### 1️⃣ Exploración de Datos
**Notebook:** `01_eda_dataset.ipynb`
- Análisis de distribución de clases
- Visualización de imágenes
- Detección de desbalance (21% sanos, 79% enfermos)
- Análisis de dimensiones y formatos

### 2️⃣ Preprocesamiento
**Notebook:** `02_test_preprocessing.ipynb`
- Implementación y test de filtros
- Comparación visual de técnicas
- Selección de mejores candidatos

### 3️⃣ Validación de Dataset
**Notebook:** `03_test_dataset.ipynb`
- Verificación de data loaders
- Test de augmentation
- Validación de preprocessing pipeline

### 4️⃣ Experimentos Baseline
**Scripts:**
- `run_baseline_none.py` - Sin filtros
- `run_baseline_ben_graham.py` - Estándar del dominio
- `run_filter_experiments.py` - 5 filtros adicionales
- `run_all_filters_experiments.sh` - Ejecución completa

**Filtros probados:**
- `none` - Sin preprocesamiento
- `ben_graham` - Estándar de la industria
- `clahe` - Contrast Limited Adaptive Histogram Equalization ⭐
- `gaussian` - Gaussian blur
- `sobel` - Detección de bordes
- `clahe_ben_graham` - Combinación
- `gaussian_clahe` - Combinación

**Mejor baseline:** `none` (84.98% val_recall)

### 5️⃣ Fine-tuning Experiments
**Scripts:**
- `run_finetuning.py` - Fine-tuning genérico con argumentos
- `run_finetuning_grid.sh` - Grid search (12 experimentos)

**Configuraciones probadas:**
- **Filtros:** none, clahe, gaussian, gaussian_clahe
- **Unfreeze values:** 80, 100, 120, 140
- **Total:** 16 experimentos (4 filtros × 4 unfreeze)
- **Learning rate:** 1e-4 (100x menor que baseline)
- **Epochs:** 30 con early stopping

**Grid Search:**
```bash
FILTERS: none, gaussian_clahe, gaussian, clahe
UNFREEZE: 140, 120, 100, 80
EPOCHS: 30
BATCH_SIZE: 32
LR: 0.0001
Total experimentos: 16
Tiempo total: ~3.5 horas
```

**Ganador:** CLAHE + unfreeze_from=120 (87.07% val_recall)

### 6️⃣ Evaluación Final
**Notebook:** `06_evaluate_test_set.ipynb`  
**Script:** `evaluate_all_models.py`

- Evaluación de 23 modelos en test set (640 imágenes)
- Comparación baseline vs fine-tuning
- Selección del mejor modelo
- Métricas finales y análisis

### 7️⃣ Análisis de Errores
**Notebook:** `07_error_analysis_by_disease.ipynb`

- Análisis por tipo de enfermedad (36 enfermedades en dataset)
- Identificación de fortalezas y debilidades
- Casos de falsos positivos/negativos
- Visualización de errores
- Recomendaciones de mejora

---

## 📈 Resultados Experimentales

### Experimentos Baseline (7 modelos)
| Experimento | Filtro | Val Recall | Val Accuracy |
|-------------|--------|------------|--------------|
| 1 | none | 84.98% | 80.16% |
| 2 | ben_graham | 81.01% | 78.75% |
| 3 | clahe | 83.28% | 78.91% |
| 4 | gaussian | 84.07% | 79.53% |
| 5 | sobel | 78.68% | 75.63% |
| 6 | clahe_ben_graham | 82.16% | 77.50% |
| 7 | gaussian_clahe | 84.62% | 80.00% |

### Fine-tuning Grid Search (16 modelos)

**Top 5 por val_recall:**
| Filtro | Unfreeze | Val Recall | Val Accuracy | Overfitting |
|--------|----------|------------|--------------|-------------|
| **clahe** | **120** | **87.07%** | **85.63%** | **1.89%** ⭐ |
| clahe | 80 | 87.25% | 83.75% | 8.20% |
| none | 120 | 86.90% | 84.06% | 6.07% |
| gaussian | 80 | 86.39% | 82.81% | 3.38% |
| gaussian_clahe | 120 | 84.62% | 80.00% | 4.18% |

### Resultados en Test Set (640 imágenes)

**Top 10 modelos:**
| Rank | Filtro | Tipo | Unfreeze | Test Recall | Test Acc | Val-Test Diff |
|------|--------|------|----------|-------------|----------|---------------|
| 🥇 | clahe | Fine-tuning | 120 | **85.18%** | **86.72%** | 1.89% ✅ |
| 🥈 | gaussian | Fine-tuning | 120 | 84.39% | 86.09% | 2.00% |
| 🥉 | gaussian | Fine-tuning | 80 | 83.00% | 85.94% | 3.38% |
| 4 | none | Fine-tuning | 120 | 80.83% | 83.75% | 6.07% |
| 5 | gaussian_clahe | Baseline | - | 80.43% | 82.66% | 4.18% |
| 6 | sobel | Baseline | - | 79.25% | 80.47% | -0.57% |
| 7 | clahe | Fine-tuning | 80 | 79.05% | 83.28% | 8.20% |
| 8 | gaussian | Baseline | - | 78.85% | 81.72% | 5.22% |
| 9 | ben_graham | Baseline | - | 77.67% | 81.09% | 3.34% |
| 10 | none | Baseline | - | 77.47% | 80.63% | 7.51% |

**Mejora de fine-tuning:** +4.75 puntos porcentuales sobre mejor baseline

---

## 🧠 Modelo Final

### Configuración
```python
Base: MobileNetV2 (ImageNet pre-trained)
Input: 224x224x3 RGB
Preprocesamiento: CLAHE

Arquitectura:
├── MobileNetV2 (154 capas)
│   ├── Capas 0-119: Congeladas (features generales)
│   └── Capas 120-154: Fine-tuned (features específicas)
├── GlobalAveragePooling2D
├── Dropout(0.3)
├── Dense(128, activation='relu')
├── Dropout(0.3)
└── Dense(2, activation='softmax')

Parámetros totales: ~2.3M
Parámetros entrenables (fine-tuning): ~800K

Optimización:
├── Baseline:
│   ├── Optimizer: Adam (lr=1e-3)
│   ├── Epochs: 30
│   └── Freeze: Toda la base MobileNetV2
└── Fine-tuning:
    ├── Optimizer: Adam (lr=1e-4)
    ├── Epochs: 30
    ├── Unfreeze from: Layer 120
    └── Early stopping: patience=8 (monitor val_recall)

Training:
├── Loss: Sparse Categorical Crossentropy
├── Metrics: Accuracy, Sparse Top-K, Custom Recall/Precision
├── Class Weights: {0: 2.39, 1: 0.63}
└── Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
```

### Preprocesamiento CLAHE
```python
1. Resize a 224x224
2. Convertir RGB → LAB
3. Aplicar CLAHE al canal L
   - clipLimit: 2.0
   - tileGridSize: (8, 8)
4. Convertir LAB → RGB
5. Normalizar a [0, 1]
```

---

## 📊 Análisis de Errores por Enfermedad

### ✅ Enfermedades Perfectamente Detectadas (Recall 100%)
- Myopia (32 casos)
- Branch Retinal Vein Occlusion (23 casos)
- Laser Scars (15 casos)
- Central Retinal Vein Occlusion (9 casos)
- Asteroid Hyalosis (5 casos)

### ✅ Excelente Detección (Recall > 90%)
- **Diabetic Retinopathy:** 96.77% (124 casos) - Objetivo principal ⭐
- **Age-Related Macular Degeneration:** 96.77% (31 casos)
- **Media Haze:** 93.27% (104 casos)

### ⚠️ Detección Problemática (Recall < 80%)
1. **Optic Disc Cupping:** 68.13% (91 casos) - Problema principal
2. **Central Serous Retinopathy:** 46.15% (13 casos)
3. **Drusens:** 73.91% (46 casos)
4. **Optic Disc Pallor:** 75.00% (24 casos)
5. **Optic Disc Edema:** 70.59% (17 casos)

**Hallazgo clave:** El modelo está optimizado para patrones vasculares (DR), pero tiene dificultad con cambios estructurales sutiles del disco óptico.

Ver análisis completo: [`docs/ERROR_ANALYSIS_REPORT.md`](docs/ERROR_ANALYSIS_REPORT.md)

---

## 🛠️ Stack Tecnológico

**Machine Learning:**
- TensorFlow 2.15 / Keras
- scikit-learn 1.3.2
- OpenCV 4.8.1
- NumPy 1.24.3, Pandas 2.1.3

**Experimentación:**
- MLflow 2.9.2 (experiment tracking)
- Jupyter Lab 4.0.9
- Matplotlib 3.8.2, Seaborn 0.13.0

**Infraestructura:**
- Docker 24.0 + Docker Compose
- NVIDIA GPU (CUDA 12.0)
- Ubuntu 24.04 LTS

**Deployment:**
- Gradio 4.44.0 (UI)
- FastAPI 0.104.1 (API)
- Uvicorn 0.24.0
- Hugging Face Spaces (hosting UI - gratis)
- Render (hosting API - free tier)

---

## 🚀 Reproducir el Proyecto

### Prerequisitos
- Docker + Docker Compose
- NVIDIA GPU + CUDA drivers (opcional, recomendado)
- 15GB espacio en disco
- 8GB RAM mínimo

### Instalación
```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/retinopathy-mle.git
cd retinopathy-mle

# 2. Descargar dataset RFMiD de Kaggle
# https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
# Extraer en: data/raw/

# 3. Construir contenedor Docker
docker-compose build

# 4. Iniciar contenedor
docker-compose up -d

# 5. Acceder a Jupyter
# http://localhost:8888
```

### Entrenar desde Cero
```bash
# Entrar al contenedor
docker exec -it retino-dev bash

# Experimentos baseline (7 modelos, ~3.5 horas)
bash scripts/run_all_filters_experiments.sh

# Fine-tuning grid search (16 modelos, ~3.5 horas)
bash scripts/run_finetuning_grid.sh

# Evaluar todos los modelos en test set
python scripts/evaluate_all_models.py

# Ver resultados en MLflow UI
mlflow ui --backend-store-uri /app/experiments/mlruns --host 0.0.0.0 --port 5000
# http://localhost:5000
```

### Entrenar Modelo Específico
```bash
# Baseline con filtro específico
python scripts/run_baseline_none.py

# Fine-tuning con configuración custom
python scripts/run_finetuning.py clahe \
  --unfreeze_from 120 \
  --epochs 30 \
  --learning_rate 0.0001 \
  --batch_size 32
```
---
## 📖 Referencias

### Dataset
- **RFMiD:** Retinal Fundus Multi-disease Image Dataset
- **Fuente:** [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
- **Paper:** Pachade S, et al. (2021). "Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research"

### Arquitectura
- **MobileNetV2:** Sandler M, et al. (2018). [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- **Transfer Learning:** Pan SJ, Yang Q. (2010). "A Survey on Transfer Learning"
- **CLAHE:** Zuiderveld K. (1994). "Contrast Limited Adaptive Histogram Equalization"

---

## ⚠️ Disclaimer

Este proyecto es **únicamente para fines educativos y demostrativos**. El modelo **NO debe utilizarse para diagnóstico médico real**. 

**Importante:**
- No reemplaza la evaluación médica profesional
- No ha sido validado clínicamente
- No está aprobado por autoridades sanitarias
- Los resultados son solo indicativos

**Siempre consulte con un oftalmólogo profesional certificado** para cualquier preocupación sobre salud ocular.

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

## 👨‍💻 Autor

**Carlos Saquel Depaoli**

<a href="https://huggingface.co/spaces/desareca/retinopathy-detector" target="_blank" style="padding-left: 20px;">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-FFD21E?style=for-the-badge&labelColor=black" alt="Hugging Face">
</a>
<a href="https://retinopathy-api.onrender.com/" target="_blank">
  <img src="https://img.shields.io/badge/Render-API-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render">
</a>

<a href="https://github.com/desareca" target="_blank">
  <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
</a>
<a href="https://linkedin.com/in/carlos-saquel" target="_blank">
  <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>
<a href="mailto:tu-correo@gmail.com">
  <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail">
</a>

---

## 📊 Mejoras Futuras

- [ ] Threshold tuning para optimizar recall/precision
- [ ] Calibración de confianza (Temperature Scaling)
- [ ] Advertencias específicas en UI para ODC
- [ ] Grad-CAM para visualización de atención del modelo
- [ ] Data augmentation específico para Optic Disc Cupping
- [ ] Multi-task learning (clasificación + segmentación)
- [ ] Ensemble con EfficientNet y ResNet50
- [ ] Multi-label classification (detectar múltiples enfermedades)
- [ ] Deploy móvil con TensorFlow Lite
- [ ] Validación clínica con oftalmólogos
- [ ] Integración con sistemas PACS hospitalarios

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub**