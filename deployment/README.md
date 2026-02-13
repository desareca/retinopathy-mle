---
title: Retinopathy Detector
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: Detector de Retinopatía Diabética usando Deep Learning
---

# 🩺 Retinopathy Detector

Detector de Retinopatía Diabética usando Deep Learning con MobileNetV2.

## 📊 Métricas del Modelo

- **Recall (Clase Enfermo):** 85.18%
- **Accuracy:** 86.72%
- **Precision:** 97.73%

## 🏗️ Arquitectura

- **Base:** MobileNetV2 (ImageNet pre-trained)
- **Preprocesamiento:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Fine-tuning:** Últimas 34 capas descongeladas (unfreeze_from=120)
- **Dataset:** RFMiD - 3,200 imágenes de retina

## 🚀 Uso

1. Sube una imagen de fondo de ojo (retina)
2. Haz click en "Analizar Imagen"
3. Obtén el resultado: Sano o Enfermo con probabilidades

## ⚠️ Disclaimer

Este modelo es solo para fines educativos y demostrativos. **NO debe usarse para diagnóstico médico real.** Consulte siempre a un profesional de la salud.

## 📝 Información Técnica

- **Framework:** TensorFlow 2.15 / Keras
- **Input:** Imágenes RGB 224x224
- **Output:** Clasificación binaria (Sano / Enfermo)
- **Preprocesamiento:** CLAHE aplicado al canal L en espacio LAB

## 👨‍💻 Desarrollado por

Carlos Saquel Depaoli

GitHub: https://github.com/desareca
