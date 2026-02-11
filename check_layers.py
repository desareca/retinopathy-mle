import tensorflow as tf

# Crear modelo base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

print("="*60)
print("ANÁLISIS DE CAPAS - MobileNetV2")
print("="*60)

print(f"\nTotal de capas en base_model: {len(base_model.layers)}")

# Contar por tipo
layer_types = {}
for layer in base_model.layers:
    layer_type = type(layer).__name__
    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

print("\nCapas por tipo:")
for layer_type, count in sorted(layer_types.items()):
    print(f"  {layer_type}: {count}")

# Mostrar primeras y últimas capas
print("\nPrimeras 10 capas:")
for i, layer in enumerate(base_model.layers[:10]):
    print(f"  [{i}] {layer.name} ({type(layer).__name__})")

print("\nÚltimas 10 capas:")
total = len(base_model.layers)
for i, layer in enumerate(base_model.layers[-10:], start=total-10):
    print(f"  [{i}] {layer.name} ({type(layer).__name__})")

# Verificar capas entrenables
print(f"\nCapas entrenables (base): {len([l for l in base_model.layers if l.trainable])}")
print(f"Capas no entrenables (base): {len([l for l in base_model.layers if not l.trainable])}")

# Crear modelo completo
from src.models.model import create_model
import sys
sys.path.append('/app')

full_model = create_model(freeze_base=False)

print("\n" + "="*60)
print("MODELO COMPLETO")
print("="*60)
print(f"Total de capas: {len(full_model.layers)}")

for i, layer in enumerate(full_model.layers):
    trainable_params = sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
    print(f"  [{i}] {layer.name} - Trainable params: {trainable_params:,}")

