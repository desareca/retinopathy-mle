"""
Evaluar todos los modelos en el test set
"""
import sys
sys.path.append('/app')

import tensorflow as tf
from pathlib import Path
import pandas as pd
import mlflow
from src.data.dataset import create_dataset

# Cargar test set
TEST_PATH = Path('/app/data/raw/Test_Set/Test_Set')
test_df = pd.read_csv(TEST_PATH / 'RFMiD_Testing_Labels.csv')

# Obtener todos los runs
runs = mlflow.search_runs(
    experiment_names=['baseline-none', 'baseline-ben-graham', 'filter-experiments']
)

results = []

for _, run in runs.iterrows():
    run_id = run['run_id']
    filter_name = run['params.filter_name']
    
    print(f"\nEvaluando: {filter_name} (run: {run_id[:8]}...)")
    
    # Cargar modelo
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.tensorflow.load_model(model_uri)
        
        # Crear test dataset
        test_dataset = create_dataset(
            df=test_df,
            img_dir=TEST_PATH / 'Test',
            batch_size=32,
            shuffle=False,
            augment=False,
            filter_name=filter_name
        )
        
        # Evaluar
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        
        results.append({
            'filter_name': filter_name,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        })
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"  Error: {str(e)}")

# Mostrar resultados
results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
print("\n" + "="*60)
print("RESULTADOS EN TEST SET")
print("="*60)
print(results_df)