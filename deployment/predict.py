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

        # Modelo Grad-CAM: salida en la última capa conv de MobileNetV2
        # 'Conv_1' es la última capa convolucional de MobileNetV2
        self._gradcam_model = self._build_gradcam_model()
    
    def _build_gradcam_model(self):
        """
        Construye modelo auxiliar para Grad-CAM.
        Resuelve la desconexión entre el grafo Sequential y la base MobileNetV2.
        """
        base_model = self.model.layers[0]  # mobilenetv2_1.00_224
        
        # Usar Conv_1_bn (post-BN) o Conv_1 — usamos out_relu para activaciones finales
        last_conv_layer_name = "Conv_1"
        
        print(f"✓ Grad-CAM usará capa: {last_conv_layer_name}")
        
        # Modelo 1: input de la BASE → (activaciones Conv_1, output de la base)
        base_grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,          # input_1
            outputs=[
                base_model.get_layer(last_conv_layer_name).output,
                base_model.output
            ]
        )
        
        # Modelo 2: output de la base → predicción final (resto del Sequential)
        # Capas después de la base: GAP, Dropout, Dense, Dropout, Dense
        self._head_layers = self.model.layers[1:]  # todo lo que sigue a MobileNetV2
        
        print(f"✓ Capas del head: {[l.name for l in self._head_layers]}")
        return base_grad_model


    def gradcam(self, image, class_id=None):
        """
        Genera heatmap Grad-CAM superpuesto sobre la imagen original.
        """
        orig_h, orig_w = image.shape[:2]

        img_processed = self.preprocess_clahe(image)
        img_batch     = np.expand_dims(img_processed, axis=0)  # (1, 224, 224, 3)

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_batch, tf.float32)
            tape.watch(img_tensor)
            
            # Pasar por base → obtener activaciones conv y output de base
            conv_outputs, base_output = self._gradcam_model(img_tensor)
            
            # Pasar output de base por el head (GAP, Dense, etc.)
            x = base_output
            for layer in self._head_layers:
                x = layer(x, training=False)
            predictions = x

            if class_id is None:
                class_id = int(tf.argmax(predictions[0]))

            class_score = predictions[:, class_id]

        # Gradientes respecto a activaciones de Conv_1
        grads       = tape.gradient(class_score, conv_outputs)  # (1, 7, 7, 1280)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (1280,)

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]  # (7, 7, 1)
        heatmap = tf.squeeze(heatmap)                               # (7, 7)
        heatmap = tf.nn.relu(heatmap).numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Redimensionar y colorear
        heatmap_resized  = cv2.resize(heatmap, (orig_w, orig_h))
        heatmap_uint8    = np.uint8(255 * heatmap_resized)
        heatmap_colored  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb      = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Imagen original uint8
        orig_img = image.copy() if image.dtype == np.uint8 else np.uint8(image * 255)

        # Superposición 60/40
        superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_rgb, 0.4, 0)
        return superimposed

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
    model_path = "/app/models/production/FINAL_clahe_ft120_best.h5"
    
    predictor = RetinopathyPredictor(model_path)
    
    print("\n✓ Predictor inicializado correctamente")
    print(f"✓ Modelo listo para hacer predicciones")
    print(f"✓ Clases: {predictor.class_names}")