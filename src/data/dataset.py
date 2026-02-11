"""
Dataset de TensorFlow para imágenes de retina
"""
import tensorflow as tf
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/app')
from src.data.preprocessing import preprocess_image, preprocess_image_with_filter


def create_dataset(df, img_dir, batch_size=32, shuffle=True, augment=False, filter_name='ben_graham'):
    """
    Crea un tf.data.Dataset
    
    Args:
        df: DataFrame con columnas 'ID' y 'Disease_Risk'
        img_dir: Directorio con las imágenes
        batch_size: Tamaño del batch
        shuffle: Si hacer shuffle
        augment: Si aplicar data augmentation
        filter_name: Filtro de preprocesamiento a aplicar
    
    Returns:
        tf.data.Dataset
    """
    img_paths = [str(Path(img_dir) / f"{img_id}.png") for img_id in df['ID'].values]
    labels = df['Disease_Risk'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    
    def load_and_preprocess(img_path, label):
        img = tf.py_function(
            func=lambda x: preprocess_image_with_filter(x, filter_name=filter_name),
            inp=[img_path],
            Tout=tf.float32
        )
        img.set_shape([224, 224, 3])
        return img, label
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    
    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset