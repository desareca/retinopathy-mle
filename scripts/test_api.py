import requests

# URL de la API
url = "http://localhost:8888/predict"

# Ruta a tu imagen
image_path = "C:/Users/carlo/Documents/Proyectos/retinopathy-mle/data/raw/Test_Set/Test_Set/Test/253.png"

# Abrir imagen y hacer request
with open(image_path, 'rb') as f:
    files = {'file': ('retina.png', f, 'image/png')}
    response = requests.post(url, files=files)

# Mostrar resultado
print("Status Code:", response.status_code)
print("\nRespuesta JSON:")
print(response.json())