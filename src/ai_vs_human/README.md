# AI vs Human Content Detection

Detección de contenido generado por IA vs escrito por humanos usando múltiples modelos de Machine Learning.

## 🎮 Soporte GPU

Este proyecto está **optimizado para GPU** y aprovechará automáticamente la aceleración por hardware cuando esté disponible.

### Modelos con GPU
- ✅ **XGBoost**: Usa `gpu_hist` para entrenamientos 5-10x más rápidos
- ✅ **PyTorch Neural Network**: Usa CUDA automáticamente para entrenamientos 10-50x más rápidos

### Modelos en CPU
- ❌ **Logistic Regression**: Solo CPU (scikit-learn)
- ❌ **Random Forest**: Solo CPU (scikit-learn)

## 📦 Instalación

### Opción 1: Con GPU (NVIDIA CUDA)

Si tienes una GPU NVIDIA con CUDA instalado:

```bash
# Instalar dependencias base
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Instalar XGBoost con GPU support
pip install xgboost

# Instalar PyTorch con CUDA 11.8 (ajusta según tu versión de CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Opción 2: Solo CPU

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

### Verificar instalación de GPU

```bash
python -c "import torch; print('PyTorch GPU:', torch.cuda.is_available())"
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
```

## 🚀 Uso

```bash
cd src/ai_vs_human
jupyter notebook ai_vs_human.ipynb
```

Ejecuta las celdas secuencialmente. El notebook detectará automáticamente si hay GPU disponible y la usará.

## 📊 Dataset

- **Archivo**: `ai_human_content_detection_dataset.csv`
- **Registros**: 14,072
- **Features**: 14 características numéricas
- **Label**: 0 = Humano, 1 = IA

## 🧠 Modelos Implementados

1. **Logistic Regression** (Baseline) - CPU
2. **XGBoost con GridSearch** - GPU/CPU
3. **Random Forest** - CPU
4. **PyTorch Neural Network** (128→64→32→1) - GPU/CPU
5. **Ensemble** (Voting Classifier) - GPU/CPU

## 📈 Resultados Esperados

- **Accuracy**: ~85-95%
- **F1-Score**: ~85-95%
- **Mejor modelo**: Generalmente XGBoost o Ensemble

## ⚡ Rendimiento GPU vs CPU

| Modelo | CPU (aprox) | GPU (aprox) | Speedup |
|--------|-------------|-------------|---------|
| XGBoost GridSearch | 5-10 min | 1-2 min | 5-10x |
| PyTorch 50 epochs | 30-60 seg | 3-10 seg | 10-50x |

## 🛠️ Requisitos del Sistema

### Para CPU
- Python 3.8+
- 8GB RAM mínimo
- 16GB RAM recomendado

### Para GPU
- Python 3.8+
- NVIDIA GPU con CUDA Compute Capability 3.5+
- CUDA 11.8 o 12.1
- 4GB+ VRAM
- 8GB+ RAM del sistema

## 📝 Notas

- El notebook funciona perfectamente sin GPU, solo será más lento
- XGBoost puede usar CPU multi-core eficientemente con `n_jobs=-1`
- PyTorch usará CPU automáticamente si no detecta GPU
