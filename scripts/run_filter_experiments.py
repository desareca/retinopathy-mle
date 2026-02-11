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


# Lista de filtros a experimentar
filters_to_test = [
    ('clahe', 'filter-experiments'),
    ('gaussian', 'filter-experiments'),
    ('sobel', 'filter-experiments'),
    ('clahe_ben_graham', 'filter-experiments'),
    ('gaussian_clahe', 'filter-experiments'),    
]

print("="*60)
print("EXPERIMENTACIÓN CON FILTROS")
print("="*60)
print(f"Total de experimentos: {len(filters_to_test)}")
print("="*60)

for idx, (filter_name, experiment_name) in enumerate(filters_to_test, 1):
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO {idx}/{len(filters_to_test)}: {filter_name.upper()}")
    print(f"{'='*60}\n")
    
    try:
        model, history = train_model(
            train_df=train_df,
            val_df=val_df,
            train_img_dir=TRAIN_PATH / 'Training',
            val_img_dir=VAL_PATH / 'Validation',
            filter_name=filter_name,
            batch_size=32,
            epochs=30,
            learning_rate=0.001,
            experiment_name=experiment_name
        )
        print(f"\n✓ {filter_name} completado exitosamente")
        
    except Exception as e:
        print(f"\n✗ Error en {filter_name}: {str(e)}")
        continue

print("\n" + "="*60)
print("✓ TODOS LOS EXPERIMENTOS COMPLETADOS")
print("="*60)
print("\nPara ver resultados:")
print("mlflow ui --backend-store-uri /app/experiments/mlruns --host 0.0.0.0 --port 5000")