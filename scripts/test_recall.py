import sys
sys.path.append('/app')
from pathlib import Path
import pandas as pd
from src.training.train import train_model

TRAIN_PATH = Path('/app/data/raw/Training_Set/Training_Set')
VAL_PATH = Path('/app/data/raw/Evaluation_Set/Evaluation_Set')

train_df = pd.read_csv(TRAIN_PATH / 'RFMiD_Training_Labels.csv')
val_df = pd.read_csv(VAL_PATH / 'RFMiD_Validation_Labels.csv')

print("="*60)
print("TEST: OPTIMIZACIÓN POR RECALL")
print("="*60)

# Test con 2 épocas
model, history = train_model(
    train_df=train_df,
    val_df=val_df,
    train_img_dir=TRAIN_PATH / 'Training',
    val_img_dir=VAL_PATH / 'Validation',
    filter_name='none',
    batch_size=32,
    epochs=2,
    learning_rate=0.001,
    experiment_name='test-recall'
)

print("\n✓ Test completado exitosamente")
