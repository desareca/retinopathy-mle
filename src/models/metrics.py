"""
Métricas personalizadas para sparse labels
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, precision_score


class SparseRecallCallback(tf.keras.callbacks.Callback):
    """Calcula recall en validación al final de cada época"""
    
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        # Obtener predicciones
        y_true = []
        y_pred = []
        
        for x_batch, y_batch in self.validation_data:
            predictions = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(y_batch.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calcular métricas
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        
        # Agregar a logs
        logs['val_recall'] = recall
        logs['val_precision'] = precision
        
        print(f" - val_recall: {recall:.4f} - val_precision: {precision:.4f}")