#!/bin/bash

API_URL="https://retinopathy-api.onrender.com"

echo "============================================================"
echo "TEST 1: Health Check"
echo "============================================================"

curl -X GET "$API_URL/health" \
  -H "accept: application/json"

echo -e "\n\n============================================================"
echo "TEST 2: Predicción"
echo "============================================================"
echo "⏳ Enviando request... (puede tardar 30-60s si es primera vez)"
echo ""

# Reemplaza 'path/to/image.png' con la ruta real
curl -X POST "$API_URL/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/retina_image.png"

echo -e "\n"