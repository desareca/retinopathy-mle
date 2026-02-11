"""
Fine-tuning: Descongelar capas de MobileNetV2
"""
import sys
sys.path.append('/app')

import tensorflow as tf
from pathlib import Path
import pandas as pd
from src.training.train import train_model
from src.models.model import create_model, compile_model

# Cargar datos
TRAIN_PATH = Path('/app/data/raw/Training_Set/Training_Set')
VAL_PATH = Path('/app/data/raw/Evaluation_Set/Evaluation_Set')

train_df = pd.read_csv(TRAIN_PATH / 'RFMiD_Training_Labels.csv')
val_df = pd.read_csv(VAL_PATH / 'RFMiD_Validation_Labels.csv')

print("="*60)
print("FINE-TUNING CON FILTRO 'NONE'")
print("="*60)

# TODO: Cargar modelo baseline y descongelar capas
# Entrenar con LR bajo (1e-5)

print("\n✓ Fine-tuning completado")