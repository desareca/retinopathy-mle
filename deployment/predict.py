"""
Función de predicción para el modelo de retinopatía
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path


class RetinopathyPredictor:
    """Predictor para modelo de retinopatía diabética"""
    
    def __init__(self, model_path):
        """
        Inicializar predictor
        
        Args:
            model_path: Ruta al modelo .h5
        """
        print(f"Cargando modelo desde: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✓ Modelo cargado exitosamente")
        
        # Clases
        self.class_names = {
            0: "Healthy (Sano)",
            1: "Disease Risk (Enfermo)"
        }
    
    def preprocess_clahe(self, image):
        """
        Aplicar preprocesamiento CLAHE a la imagen
        
        Args:
            image: Imagen numpy array (H, W, 3)
            
        Returns:
            Imagen preprocesada (224, 224, 3) normalizada [0, 1]
        """
        # Resize a 224x224
        img_resized = cv2.resize(image, (224, 224))
        
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
    
    def predict(self, image):
        """
        Hacer predicción en una imagen
        
        Args:
            image: Imagen como numpy array (H, W, 3) en RGB
            
        Returns:
            dict con:
                - class_id: int (0 o 1)
                - class_name: str
                - confidence: float
                - probabilities: dict {class_name: probability}
        """
        # Preprocesar
        img_processed = self.preprocess_clahe(image)
        
        # Agregar batch dimension
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Predecir
        predictions = self.model.predict(img_batch, verbose=0)
        
        # Extraer resultados
        class_id = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_id])
        
        # Crear resultado
        result = {
            'class_id': class_id,
            'class_name': self.class_names[class_id],
            'confidence': confidence,
            'probabilities': {
                self.class_names[0]: float(predictions[0][0]),
                self.class_names[1]: float(predictions[0][1])
            }
        }
        
        return result


# Función de prueba
if __name__ == "__main__":
    # Probar con el modelo
    model_path = "/app/models/production/FINAL_clahe_ft120_best.h5"
    
    predictor = RetinopathyPredictor(model_path)
    
    print("\n✓ Predictor inicializado correctamente")
    print(f"✓ Modelo listo para hacer predicciones")
    print(f"✓ Clases: {predictor.class_names}")