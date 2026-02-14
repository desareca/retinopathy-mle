/**
 * Ejemplo de uso de la API desde JavaScript/Node.js
 */

const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const API_URL = 'https://retinopathy-api.onrender.com';

// Test 1: Health Check
async function testHealth() {
    console.log('='.repeat(60));
    console.log('TEST 1: Health Check');
    console.log('='.repeat(60));
    
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();
    
    console.log(`Status: ${response.status}`);
    console.log('Response:', JSON.stringify(data, null, 2));
    console.log();
}

// Test 2: Predicción
async function testPredict(imagePath) {
    console.log('='.repeat(60));
    console.log('TEST 2: Predicción');
    console.log('='.repeat(60));
    console.log(`Imagen: ${imagePath}`);
    console.log('⏳ Enviando request... (puede tardar 30-60s si es primera vez)');
    
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: form
    });
    
    const data = await response.json();
    
    console.log(`Status: ${response.status}`);
    
    if (response.status === 200) {
        console.log('\n✓ Predicción exitosa:');
        console.log(`  - Resultado: ${data.prediction}`);
        console.log(`  - Confianza: ${(data.confidence * 100).toFixed(1)}%`);
        console.log(`  - Probabilidades:`);
        console.log(`    • Sano: ${(data.probabilities.Healthy * 100).toFixed(1)}%`);
        console.log(`    • Enfermo: ${(data.probabilities.Disease * 100).toFixed(1)}%`);
        console.log(`\n  ${data.message}`);
    } else {
        console.log(`✗ Error: ${JSON.stringify(data)}`);
    }
    console.log();
}

// Ejecutar tests
async function main() {
    await testHealth();
    
    // Reemplaza con la ruta a tu imagen
    const imagePath = 'path/to/your/retina_image.png';
    
    try {
        await testPredict(imagePath);
    } catch (error) {
        console.log(`⚠️  Error: ${error.message}`);
        console.log('   Descarga una imagen de retina y actualiza la ruta');
    }
}

main();