#!/bin/bash

echo "=========================================="
echo "GRID SEARCH: FINE-TUNING"
echo "=========================================="

# Configuración
EPOCHS=20
BATCH_SIZE=32

# Filtros a probar (basados en mejores resultados baseline)
FILTERS=("none" "gaussian_clahe" "gaussian" "clahe")

# Valores de unfreeze_from a probar
UNFREEZE_VALUES=(150 125 100 75)

# Contadores
TOTAL_EXPERIMENTS=$((${#FILTERS[@]} * ${#UNFREEZE_VALUES[@]}))
CURRENT=0
START_TOTAL=$(date +%s)

echo ""
echo "Configuración:"
echo "  Filtros: ${FILTERS[@]}"
echo "  Unfreeze values: ${UNFREEZE_VALUES[@]}"
echo "  Épocas: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Total experimentos: $TOTAL_EXPERIMENTS"
echo "  Tiempo estimado: ~2.5 horas"
echo "=========================================="

# Log file
LOG_FILE="/app/experiments/finetuning_grid_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

# Función para estimar tiempo
estimate_time() {
    local unfreeze=$1
    if [ $unfreeze -ge 150 ]; then
        echo "8"
    elif [ $unfreeze -ge 125 ]; then
        echo "12"
    elif [ $unfreeze -ge 100 ]; then
        echo "16"
    else
        echo "20"
    fi
}

# Loop principal
for filter in "${FILTERS[@]}"
do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "FILTRO: $filter"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for unfreeze in "${UNFREEZE_VALUES[@]}"
    do
        CURRENT=$((CURRENT + 1))
        ESTIMATED_TIME=$(estimate_time $unfreeze)
        
        echo ""
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Experimento:"
        echo "  Filtro: $filter"
        echo "  Unfreeze from: $unfreeze"
        echo "  Tiempo estimado: ~$ESTIMATED_TIME min"
        echo "  Hora inicio: $(date '+%H:%M:%S')"
        
        START_TIME=$(date +%s)
        
        # Ejecutar fine-tuning
        python /app/scripts/run_finetuning.py $filter \
            --unfreeze_from $unfreeze \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            2>&1 | tee -a $LOG_FILE
        
        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✓ Completado exitosamente"
            echo "  Duración real: $((DURATION/60)) min $((DURATION%60)) sec"
        else
            echo "  ✗ Error en experimento"
            echo "  Revisar log: $LOG_FILE"
        fi
        
        # Tiempo restante estimado
        REMAINING=$((TOTAL_EXPERIMENTS - CURRENT))
        if [ $REMAINING -gt 0 ]; then
            AVG_TIME=$((DURATION / 60))
            EST_REMAINING=$((AVG_TIME * REMAINING))
            echo "  Experimentos restantes: $REMAINING"
            echo "  Tiempo restante estimado: ~$EST_REMAINING min"
        fi
    done
done

END_TOTAL=$(date +%s)
TOTAL_DURATION=$((END_TOTAL - START_TOTAL))

echo ""
echo "=========================================="
echo "✓ GRID SEARCH COMPLETADO"
echo "=========================================="
echo "Total experimentos: $TOTAL_EXPERIMENTS"
echo "Duración total: $((TOTAL_DURATION/60)) min $((TOTAL_DURATION%60)) sec"
echo "Promedio por experimento: $((TOTAL_DURATION/TOTAL_EXPERIMENTS/60)) min"
echo "Log guardado en: $LOG_FILE"
echo ""
echo "Para ver resultados:"
echo "  mlflow ui --backend-store-uri /app/experiments/mlruns --host 0.0.0.0 --port 5000"
echo "=========================================="