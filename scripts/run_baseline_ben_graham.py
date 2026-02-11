"""
Baseline 2: Ben Graham (estándar del dominio)
"""
import sys
sys.path.append('/app')

from pathlib import Path
import pandas as pd
from src.training.train import train_model

# Cargar datos
TRAIN_PATH = Path('/app/data/raw/Training_Set/Training_Set')
VAL_PATH = Path('/app/data/raw/Evaluation_Set/Evaluation_Set')

train_df = pd.read_csv(TRAIN_PATH / 'RFMiD_Training_Labels.csv')
val_df = pd.read_csv(VAL_PATH / 'RFMiD_Validation_Labels.csv')

print("="*60)
print("BASELINE 2: BEN GRAHAM")
print("="*60)

# Entrenar
model, history = train_model(
    train_df=train_df,
    val_df=val_df,
    train_img_dir=TRAIN_PATH / 'Training',
    val_img_dir=VAL_PATH / 'Validation',
    filter_name='ben_graham',
    batch_size=32,
    epochs=30,
    learning_rate=0.001,
    experiment_name='baseline-ben-graham'
)

print("\n✓ Baseline Ben Graham completado")