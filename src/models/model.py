"""
Modelo MobileNetV2 para clasificación de retinopatía
"""
import tensorflow as tf


def create_model(input_shape=(224, 224, 3), num_classes=2, freeze_base=True):
    """
    Crea modelo MobileNetV2 con Transfer Learning
    
    Args:
        input_shape: Tamaño de entrada
        num_classes: Número de clases (2 para binario)
        freeze_base: Si congelar la base MobileNetV2
    
    Returns:
        Modelo sin compilar
    """
    # Base pre-entrenada
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = not freeze_base
    
    # Custom head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila el modelo
    
    Args:
        model: Modelo a compilar
        learning_rate: Learning rate
    
    Returns:
        Modelo compilado
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
        ]
    )
    
    return model