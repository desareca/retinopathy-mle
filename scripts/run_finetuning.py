"""
Fine-tuning genérico: Descongelar capas de MobileNetV2
Uso: python scripts/run_finetuning.py <filter_name>
Ejemplo: python scripts/run_finetuning.py none
"""
import sys
sys.path.append('/app')

import argparse
import tensorflow as tf
from pathlib import Path
import pandas as pd
import mlflow
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime

from src.data.dataset import create_dataset
from src.models.model import create_model
from src.models.metrics import SparseRecallCallback


def run_finetuning(
    filter_name='none',
    epochs=20,
    batch_size=32,
    learning_rate=1e-5,
    unfreeze_from_layer=100
):
    """
    Ejecuta fine-tuning con el filtro especificado
    
    Args:
        filter_name: Filtro a usar (none, ben_graham, clahe, etc.)
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Learning rate (usar 1e-5 para fine-tuning)
        unfreeze_from_layer: Desde qué capa descongelar (0=todas, 100=últimas capas)
    """
    
    # Cargar datos
    TRAIN_PATH = Path('/app/data/raw/Training_Set/Training_Set')
    VAL_PATH = Path('/app/data/raw/Evaluation_Set/Evaluation_Set')
    
    train_df = pd.read_csv(TRAIN_PATH / 'RFMiD_Training_Labels.csv')
    val_df = pd.read_csv(VAL_PATH / 'RFMiD_Validation_Labels.csv')
    
    print("="*60)
    print(f"FINE-TUNING CON FILTRO '{filter_name.upper()}'")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Unfreeze from layer: {unfreeze_from_layer}")
    print("="*60)
    
    # Configurar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # MLflow setup
    mlflow.set_tracking_uri('file:///app/experiments/mlruns')
    mlflow.set_experiment('finetuning')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{filter_name}_ft_ep{epochs}_lr{learning_rate}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        
        # Log parameters
        mlflow.log_param("filter_name", filter_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("fine_tuning", True)
        mlflow.log_param("unfreeze_from_layer", unfreeze_from_layer)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        
        # Calcular class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['Disease_Risk'].values),
            y=train_df['Disease_Risk'].values
        )
        class_weights = dict(zip([0, 1], class_weights))
        print(f"\nClass weights: {class_weights}")
        mlflow.log_param("class_weight_0", class_weights[0])
        mlflow.log_param("class_weight_1", class_weights[1])
        
        # Crear datasets
        print("\nCreando datasets...")
        train_dataset = create_dataset(
            train_df, TRAIN_PATH / 'Training',
            batch_size=batch_size, shuffle=True, augment=True, filter_name=filter_name
        )
        
        val_dataset = create_dataset(
            val_df, VAL_PATH / 'Validation',
            batch_size=batch_size, shuffle=False, augment=False, filter_name=filter_name
        )
        
        # Crear modelo
        print("\nCreando modelo con fine-tuning...")
        model = create_model(freeze_base=False)  # No congelar base
        
        # Congelar selectivamente
        base_model = model.layers[0]
        total_layers = len(base_model.layers)
        
        for i, layer in enumerate(base_model.layers):
            if i < unfreeze_from_layer:
                layer.trainable = False
            else:
                layer.trainable = True
        
        trainable_layers = len([l for l in model.layers if l.trainable])
        frozen_layers = len([l for l in model.layers if not l.trainable])
        
        print(f"\nTotal capas base: {total_layers}")
        print(f"Capas congeladas: {frozen_layers}")
        print(f"Capas entrenables: {trainable_layers}")
        
        # Compilar con LR muy bajo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_accuracy')]
        )
        
        # Callbacks
        experiment_id = f"finetuning_{filter_name}_{timestamp}"
        checkpoint_path = f'/app/models/checkpoints/{experiment_id}_best.h5'
        final_model_path = f'/app/models/production/{experiment_id}_final.h5'
        
        recall_callback = SparseRecallCallback(val_dataset)
        
        callbacks = [
            recall_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall',
                patience=8,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_recall',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_recall',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Entrenar
        print("\nIniciando fine-tuning...\n")
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
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_recall': history.history['val_recall'][-1],
            'val_precision': history.history['val_precision'][-1],
        }
        
        for metric, value in final_metrics.items():
            mlflow.log_metric(metric, value)
        
        # Guardar modelo
        model.save(final_model_path)
        mlflow.tensorflow.log_model(model, "model")
        
        mlflow.log_param("checkpoint_path", checkpoint_path)
        mlflow.log_param("final_model_path", final_model_path)
        
        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETADO")
        print(f"{'='*60}")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"{'='*60}")
        print(f"\nModelos guardados:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Final: {final_model_path}")
        print(f"{'='*60}")
        
        return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning de MobileNetV2')
    
    parser.add_argument('filter_name', type=str, 
                       help='Filtro a usar (none, ben_graham, clahe, gaussian, etc.)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Número de épocas (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--unfreeze_from', type=int, default=100,
                       help='Desde qué capa descongelar (default: 100)')
    
    args = parser.parse_args()
    
    run_finetuning(
        filter_name=args.filter_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        unfreeze_from_layer=args.unfreeze_from
    )
    
    print("\n✓ Fine-tuning completado exitosamente")