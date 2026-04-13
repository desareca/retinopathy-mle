"""
Demo interactiva de Retinopathy Detector con Gradio
"""
import gradio as gr
import numpy as np
from PIL import Image
import cv2
from predict import RetinopathyPredictor
import matplotlib.pyplot as plt
import io

# Inicializar predictor
MODEL_PATH = "model.h5"
predictor = RetinopathyPredictor(MODEL_PATH)

def cambiar_icono(icono_actual):
    # Si detecta el símbolo de sol, cambia a luna, y viceversa
    if icono_actual == "☼":
        return gr.update(value="⏾")
    return gr.update(value="☼")

def predict_image(image):
    """
    Función que se llama cuando el usuario sube una imagen
    
    Args:
        image: PIL Image o numpy array
        
    Returns:
        tuple: (texto_resultado, diccionario_probabilidades, imagen_comparacion)
    """
    if image is None:
        return "⚠️ Por favor sube una imagen primero", None, None
    
    # Convertir a numpy si es PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Asegurar que sea RGB
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Crear imagen procesada para comparación
    img_resized = cv2.resize(image, (224, 224))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Crear comparación
    #comparison = create_comparison_image(img_resized, img_processed)
    
    # Hacer predicción
    result = predictor.predict(image)
    print(result)
    
    # Formatear resultado
    class_name = result['class_name']
    confidence = result['confidence'] * 100
    
    # Emoji y color según resultado
    if result['class_id'] == 0:
        emoji = "✅"
        color = "#28a745"  # Verde
        recommendation = "**✓ Imagen sugiere retina saludable.** Mantener controles regulares."
    else:
        emoji = "⚠️"
        color = "#dc3545"  # Rojo
        recommendation = "**! Se detectaron posibles signos de riesgo.** Consulte con un oftalmólogo profesional."
    
    # Texto de resultado mejorado
    resultado_texto = f"""
    <div style="padding: 5px 15px 15px 15px; border-left: 4px solid {color}; background: linear-gradient(to right, {color}15, transparent);">
    
    ## {emoji} Resultado: {class_name}
    
    ### 📊 Confianza: {confidence:.1f}%
    
    **Probabilidades Detalladas:**
    - 🟢 **Sano (Healthy):** {result['probabilities']['Healthy (Sano)'] * 100:.2f}%
    - 🔴 **Enfermo (Disease Risk):** {result['probabilities']['Disease Risk (Enfermo)'] * 100:.2f}%
  
    {recommendation}
    
    <small style="color: #6c757d;">
    *Nota: Modelo con 85.18% recall - Detecta 85 de cada 100 casos con riesgo*
    </small>
    
    </div>
    """
    
    # Diccionario para el gráfico de barras
    probabilidades = {
        "Sano": result['probabilities']['Healthy (Sano)'],
        "Enfermo": result['probabilities']['Disease Risk (Enfermo)']
    }
    
    return resultado_texto#, probabilidades#, comparison


# Crear interfaz

# Definimos un JS pequeño para activar el cambio de tema
toggle_theme_js = """
function() {
    document.querySelector('body').classList.toggle('dark');
}
"""

# theme = gr.themes.Glass(
theme = gr.themes.Soft(
    #primary_hue="lime", 
    #secondary_hue="lime",
    font=[gr.themes.GoogleFont("Open Sans"), "Arial", "sans-serif"]
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
    background: #7192ff;
}
/* Botón transparente */
.btn-transparente {
    background: none !important;
    background-color: transparent !important;
    border: 1px solid black !important;
    border-radius: 10% !important;
    box-shadow: none !important;
    width: 50px !important;

    margin-left: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    /* MODO CLARO: Color Negro */
    color: black !important;
    font-size: 20px !important;
    transition: all 0.3s ease !important;
}

/* MODO OSCURO: Color Blanco */
.dark .btn-transparente {
    color: white !important;
    border: 1px solid white !important;
}

/* Efecto hover sutil */
.btn-transparente:hover {
    transform: scale(1.1);
    background: transparent !important;
}

.col-izq {
    padding: 5px 15px 15px 15px;
    border-left: 4px solid #7192ff;
    background: linear-gradient(to right, #7192ff15, transparent);
}

.disclaimer_message {
    padding: 5px 15px 7px 15px;
    border-left: 4px solid #ffb85b;
    background: linear-gradient(to right, #ffb85b30, transparent);
}

.autor {
    text-align: center;
}
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=20): # El título ocupa la mayoría del espacio
            gr.Markdown("""
                # 🩺 Retinopathy Detector
                
                **Detector de Retinopatía Diabética usando Deep Learning**
                
                Este modelo usa MobileNetV2 con fine-tuning y preprocesamiento CLAHE para detectar 
                riesgo de enfermedad en imágenes de fondo de ojo.

                """)
        with gr.Column(scale=1, min_width="auto"): # El botón se queda en la esquina
            theme_btn = gr.Button("☼", variant="secondary", elem_classes="btn-transparente")

    gr.Markdown("""---""")

   
    # Título y descripción

    
    with gr.Row():
        # Columna izquierda: Input
        with gr.Column(scale=27, elem_classes="col-metrics"):
            gr.Markdown("""
                ## 📊 Métricas del Modelo:
                - **Recall:** 85.18% (detecta 85 de cada 100 casos enfermos)
                - **Accuracy:** 86.72%
                - **Precision:** 97.73%
                
                ## 📝 Instrucciones:
                1. Sube una imagen de retina (formato: PNG, JPG, JPEG)
                2. Haz click en "🔍 Analizar Imagen"
                3. Ve el resultado y las probabilidades
                                      
                ## 💡 Consejos:
                - Usa imágenes claras de fondo de ojo
                - Formato recomendado: PNG o JPG
                - La imagen se redimensiona automáticamente
                """,
                elem_classes=["col-izq"])
            
        with gr.Column(scale=43, elem_classes="col-upload-image"):
            input_image = gr.Image(
                label="Subir Imagen de Retina",
                sources=["upload"],
                type="numpy",
                height=400
            )
            
            predict_btn = gr.Button(
                "🔍 Analizar Imagen",
                elem_classes="button-upload-image",
                variant="primary",
                # size="lg"
            )

        # Columna derecha: Output
        with gr.Column(scale=30, elem_classes="col-results"):
            output_text = gr.Markdown(label="Resultado")
            
            #output_plot = gr.Label(
            #    label="Probabilidades",
            #    show_label=True,
            #    num_top_classes=2
            #)
    
    # Conectar botón con función
    predict_btn.click(
        fn=predict_image,
        inputs=[input_image],
        outputs=[output_text]#, output_plot]
    )

    theme_btn.click(None, None, None, js=toggle_theme_js)
    theme_btn.click(fn=cambiar_icono, inputs=[theme_btn], outputs=[theme_btn])

    
    # Footer
    gr.Markdown("""
    ---
    
    ### ℹ️ Información del Modelo
    
    - **Arquitectura:** MobileNetV2 + Transfer Learning
    - **Preprocesamiento:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - **Dataset:** [RFMiD (3,200 imágenes)](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
    - **Framework:** TensorFlow/Keras               
    """)

    gr.Markdown("""
    **⚠️ Disclaimer:** Este modelo es solo para fines educativos y demostrativos. 
    No debe usarse para diagnóstico médico real. Consulte a un profesional de la salud.
    """,
    elem_classes=["disclaimer_message"])

    gr.Markdown("""
    ---                   
    """)

    gr.Markdown("""
    Desarrollado por Carlos Saquel Depaoli        
    """,
    elem_classes=["autor"])

# Lanzar aplicación
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )