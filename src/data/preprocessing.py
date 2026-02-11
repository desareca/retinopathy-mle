"""
Funciones de preprocesamiento de imágenes de retina
"""
import cv2
import numpy as np


def crop_image_from_gray(img, tol=7):
    """Recorta bordes negros"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img, sigmaX=10):
    """Crop circular + Ben Graham"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
    return img


def preprocess_image(img_path, target_size=(224, 224)):
    """
    Pipeline completo de preprocesamiento
    """
    # Decodificar el tensor string a string de Python
    if hasattr(img_path, 'numpy'):
        img_path = img_path.numpy().decode('utf-8')
    else:
        img_path = str(img_path)
    
    # Leer imagen
    img = cv2.imread(img_path)
    
    # Verificar que la imagen se cargó
    if img is None:
        # Si falla, retornar imagen negra
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    
    # Circle crop (Ben Graham)
    img = circle_crop(img)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalizar (0-1)
    img = img.astype(np.float32) / 255.0
    
    return img

def apply_clahe(img):
    """Aplica CLAHE para mejorar contraste"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """Aplica Gaussian Blur"""
    return cv2.GaussianBlur(img, kernel_size, 0)


def apply_sobel(img):
    """Aplica filtro Sobel para detección de bordes"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)


def preprocess_image_with_filter(img_path, target_size=(224, 224), filter_name='ben_graham'):
    """
    Pipeline de preprocesamiento con filtro seleccionable
    
    Args:
        img_path: Ruta a la imagen
        target_size: Tamaño objetivo
        filter_name: Nombre del filtro a aplicar
                    Opciones: 'none', 'ben_graham', 'clahe', 'gaussian', 'sobel', 
                             'clahe_ben_graham', 'gaussian_clahe'
    
    Returns:
        Imagen preprocesada
    """
    # Decodificar path
    if hasattr(img_path, 'numpy'):
        img_path = img_path.numpy().decode('utf-8')
    else:
        img_path = str(img_path)
    
    # Leer imagen
    img = cv2.imread(img_path)
    
    if img is None:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    
    # Convertir a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop básico
    img = crop_image_from_gray(img)
    
    # Aplicar filtro según experimento
    if filter_name == 'none':
        # Solo crop y resize
        pass
    
    elif filter_name == 'ben_graham':
        img = circle_crop(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    elif filter_name == 'clahe':
        img = apply_clahe(img)
    
    elif filter_name == 'gaussian':
        img = apply_gaussian_blur(img)
    
    elif filter_name == 'sobel':
        img = apply_sobel(img)
    
    elif filter_name == 'clahe_ben_graham':
        img = circle_crop(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = apply_clahe(img)
    
    elif filter_name == 'gaussian_clahe':
        img = apply_gaussian_blur(img)
        img = apply_clahe(img)
    
    else:
        raise ValueError(f"Filtro '{filter_name}' no reconocido")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalizar
    img = img.astype(np.float32) / 255.0
    
    return img
