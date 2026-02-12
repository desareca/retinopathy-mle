"""
Demo interactiva de Retinopathy Detector con Gradio
"""
import gradio as gr
import numpy as np
from PIL import Image
from predict import RetinopathyPredictor


# Inicializar predictor
MODEL_PATH = "/app/models/production/FINAL_clahe_ft120_best.h5"
predictor = RetinopathyPredictor(MODEL_PATH)


def predict_image(image):
    """
    Función que se llama cuando el usuario sube una imagen
    
    Args:
        image: PIL Image o numpy array
        
    Returns:
        tuple: (texto_resultado, diccionario_probabilidades)
    """
    # Convertir a numpy si es PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Asegurar que sea RGB
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Hacer predicción
    result = predictor.predict(image)
    
    # Formatear resultado
    class_name = result['class_name']
    confidence = result['confidence'] * 100
    
    # Emoji según resultado
    emoji = "✅" if result['class_id'] == 0 else "⚠️"
    
    # Texto de resultado
    resultado_texto = f"""
    ## {emoji} Resultado: {class_name}
    
    **Confianza:** {confidence:.1f}%
    
    ### Probabilidades:
    - **Sano (Healthy):** {result['probabilities']['Healthy (Sano)'] * 100:.1f}%
    - **Enfermo (Disease Risk):** {result['probabilities']['Disease Risk (Enfermo)'] * 100:.1f}%
    """
    
    # Diccionario para el gráfico de barras
    probabilidades = {
        "Sano": result['probabilities']['Healthy (Sano)'],
        "Enfermo": result['probabilities']['Disease Risk (Enfermo)']
    }
    
    return resultado_texto, probabilidades


# Crear interfaz
theme = gr.themes.Ocean(
    #primary_hue="blue", 
    #secondary_hue="lime",
    font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"]
)

custom_css = """
.col-upload-image {
    padding: 0px 10px 0px 10px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center;
}
.col-metrics {
    padding: 0px 10px 0px 20px !important; 
}
.col-results {
    padding: 0px 0px 0px 10px !important; 
}
.button-upload-image {
    width: 200px !important;
}
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    
    # Título y descripción
    gr.Markdown("""
    # 🩺 Retinopathy Detector
    
    **Detector de Retinopatía Diabética usando Deep Learning**
    
    Este modelo usa MobileNetV2 con fine-tuning y preprocesamiento CLAHE para detectar 
    riesgo de enfermedad en imágenes de fondo de ojo.
                
    ---
    """)
    
    with gr.Row():
        # Columna izquierda: Input
        with gr.Column(scale=0.3, elem_classes="col-metrics"):
            output_text = gr.Markdown("""
                ### 📊 Métricas del Modelo:
                - **Recall:** 85.18% (detecta 85 de cada 100 casos enfermos)
                - **Accuracy:** 86.72%
                - **Precision:** 97.73%
                
                ### 📝 Instrucciones:
                1. Sube una imagen de retina (formato: PNG, JPG, JPEG)
                2. Haz click en "🔍 Analizar Imagen"
                3. Ve el resultado y las probabilidades
                                      
                ### 💡 Consejos:
                - Usa imágenes claras de fondo de ojo
                - Formato recomendado: PNG o JPG
                - La imagen se redimensiona automáticamente
                """)
            
        with gr.Column(scale=0.4, elem_classes="col-upload-image"):
            input_image = gr.Image(
                label="📤 Subir Imagen de Retina",
                sources=["upload"],
                type="numpy",
                height=400
            )
            
            predict_btn = gr.Button(
                "🔍 Analizar Imagen",
                elem_classes="button-upload-image",
                variant="primary",
                size="lg"
            )

        # Columna derecha: Output
        with gr.Column(scale=0.3, elem_classes="col-results"):
            output_text = gr.Markdown(label="Resultado")
            
            output_plot = gr.Label(
                label="Probabilidades",
                show_label=False,
                num_top_classes=2
            )
    
    # Conectar botón con función
    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_text, output_plot]
    )

    
    # Footer
    gr.Markdown("""
    ---
    
    ### ℹ️ Información del Modelo
    
    - **Arquitectura:** MobileNetV2 + Transfer Learning
    - **Preprocesamiento:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - **Dataset:** RFMiD (3,200 imágenes)
    - **Framework:** TensorFlow/Keras
    
    **⚠️ Disclaimer:** Este modelo es solo para fines educativos y demostrativos. 
    No debe usarse para diagnóstico médico real. Consulte a un profesional de la salud.
    
    ---
    
    Desarrollado por: [Tu Nombre]
    """)


# Lanzar aplicación
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )