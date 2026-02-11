# Diabetic Retinopathy Detection - MLE Project

## 🎯 Objetivo
Clasificación binaria de imágenes de retina: Sano (0) vs Enfermo (1)

## 📊 Dataset
- **Fuente:** RFMiD (Kaggle)
- **Total:** 3,200 imágenes
- **Train:** 1,920 | **Val:** 640 | **Test:** 640
- **Desbalance:** 21% sanos, 79% enfermos

## 🏆 Resultados Actuales

### Baselines (30 épocas)
| Filtro | Train Acc | Val Acc | Duración | Observaciones |
|--------|-----------|---------|----------|---------------|
| None | 80.78% | **80.47%** | 9.6 min | ✓ Mejor baseline |
| Ben Graham | 75.83% | 79.37% | 28.2 min | Más lento, peor resultado |

### Experimentos en curso
- [ ] CLAHE
- [ ] Gaussian Blur
- [ ] Gaussian + CLAHE
- [ ] Sobel

## 🏗️ Arquitectura
- **Base:** MobileNetV2 (ImageNet pre-trained, congelado)
- **Head:** GAP → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(2)
- **Optimizador:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Class Weights:** Balanceados automáticamente

## 🚀 Próximos Pasos
1. ✅ Baseline None y Ben Graham
2. 🔄 Experimentación con filtros
3. ⏳ Fine-tuning (descongelar capas)
4. ⏳ Ajuste de threshold
5. ⏳ Deployment (API + Gradio)