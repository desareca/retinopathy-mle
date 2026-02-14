"""
Ejemplos de uso de la Retinopathy Detector API
"""
import requests
import json

# URL de la API
API_URL = "https://retinopathy-api.onrender.com"

def test_health():
    """Verificar que la API está funcionando"""
    print("="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict(image_path):
    """Hacer predicción con una imagen"""
    print("="*60)
    print("TEST 2: Predicción")
    print("="*60)
    print(f"Imagen: {image_path}")
    print("⏳ Enviando request... (puede tardar 30-60s si es primera vez)")
    
    with open(image_path, 'rb') as f:
        files = {'file': ('retina.png', f, 'image/png')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Predicción exitosa:")
        print(f"  - Resultado: {result['prediction']}")
        print(f"  - Confianza: {result['confidence']*100:.1f}%")
        print(f"  - Probabilidades:")
        print(f"    • Sano: {result['probabilities']['Healthy']*100:.1f}%")
        print(f"    • Enfermo: {result['probabilities']['Disease']*100:.1f}%")
        print(f"\n  {result['message']}")
    else:
        print(f"✗ Error: {response.text}")
    print()


if __name__ == "__main__":
    # Test 1: Health check
    test_health()
    
    # Test 2: Predicción
    # Reemplaza con la ruta a tu imagen
    image_path = "path/to/your/retina_image.png"
    
    try:
        test_predict(image_path)
    except FileNotFoundError:
        print(f"⚠️  No se encontró la imagen: {image_path}")
        print("   Descarga una imagen de retina y actualiza la ruta")