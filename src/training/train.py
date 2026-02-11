"""
Script de entrenamiento del modelo
"""
import sys
sys.path.append('/app')

import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import mlflow
import mlflow.tensorflow
from datetime import datetime

from src.data.dataset import create_dataset
from src.models.model import create_model, compile_model


def calculate_class_weights(labels):
    """Calcula pesos para clases desbalanceadas"""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def train_model(
    train_df,
    val_df,
    train_img_dir,
    val_img_dir,
    filter_name='ben_graham',
    batch_size=32,
    epochs=30,
    learning_rate=0.001,
    experiment_name='retinopathy-baseline'
):
    """
    Entrena el modelo
    
    Args:
        train_df: DataFrame de entrenamiento
        val_df: DataFrame de validación
        train_img_dir: Directorio de imágenes de train
        val_img_dir: Directorio de imágenes de validación
        filter_name: Filtro de preprocesamiento
        batch_size: Tamaño del batch
        epochs: Número de épocas
        learning_rate: Learning rate
        experiment_name: Nombre del experimento en MLflow
    """
    
    # Configurar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # MLflow setup
    mlflow.set_tracking_uri('file:///app/experiments/mlruns')
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{filter_name}_ep{epochs}_bs{batch_size}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("filter_name", filter_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        
        # Calcular class weights
        class_weights = calculate_class_weights(train_df['Disease_Risk'].values)
        print(f"Class weights: {class_weights}")
        mlflow.log_param("class_weight_0", class_weights[0])
        mlflow.log_param("class_weight_1", class_weights[1])
        
        # Crear datasets
        print("Creando datasets...")
        train_dataset = create_dataset(
            train_df, train_img_dir, 
            batch_size=batch_size, 
            shuffle=True, 
            augment=True,
            filter_name=filter_name
        )
        
        val_dataset = create_dataset(
            val_df, val_img_dir,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            filter_name=filter_name
        )
        
        # Crear modelo
        print("Creando modelo...")
        model = create_model(freeze_base=True)
        model = compile_model(model, learning_rate=learning_rate)
        
        print(model.summary())
        
        # Callbacks
        experiment_id = f"{experiment_name}_{filter_name}_{timestamp}"
        
        checkpoint_path = f'/app/models/checkpoints/{experiment_id}_best.h5'
        final_model_path = f'/app/models/production/{experiment_id}_final.h5'

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        print("Iniciando entrenamiento...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log metrics finales
        final_metrics = {
            'train_loss': history.history['loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
        
        for metric, value in final_metrics.items():
            mlflow.log_metric(metric, value)
        
        # Guardar modelo con nombre único
        model.save(final_model_path)
        mlflow.tensorflow.log_model(model, "model")
        
        # Log de paths
        mlflow.log_param("checkpoint_path", checkpoint_path)
        mlflow.log_param("final_model_path", final_model_path)
        
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"{'='*60}")
        
        return model, history


if __name__ == "__main__":
    # Cargar datos
    TRAIN_PATH = Path('/app/data/raw/Training_Set/Training_Set')
    VAL_PATH = Path('/app/data/raw/Evaluation_Set/Evaluation_Set')
    
    train_df = pd.read_csv(TRAIN_PATH / 'RFMiD_Training_Labels.csv')
    val_df = pd.read_csv(VAL_PATH / 'RFMiD_Validation_Labels.csv')
    
    # Entrenar
    model, history = train_model(
        train_df=train_df,
        val_df=val_df,
        train_img_dir=TRAIN_PATH / 'Training',
        val_img_dir=VAL_PATH / 'Validation',
        filter_name='ben_graham',
        batch_size=32,
        epochs=2,  # Solo 5 para prueba rápida
        learning_rate=0.001
    )