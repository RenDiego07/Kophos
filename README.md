# Kophos

Kophos is a mobile app that's meant to be used to shrink the speaking barriers between deaf and hearing communities. It also intends to help non sign language speakers understand basic sign language with the help of deep neural networks (CNN and LSTM).

## Descripción del Proyecto

Este proyecto implementa un sistema de reconocimiento de lenguaje de señas americano (ASL) utilizando:
- **MobileNetV2** para extracción de características visuales
- **BiLSTM** (Bidirectional LSTM) para modelado de secuencias temporales
- **Attention Mechanism** para enfocarse en frames importantes
- Dataset **NSLT-100** (subconjunto de WLASL con 100 clases)

## Estructura del Proyecto

```
Kophos/
├── data/
│   ├── raw/                    # Archivos originales (IGNORAR EN GIT)
│   │   ├── WLASL_v0.3.json    # Metadatos completos de WLASL
│   │   ├── nslt_100.json      # Subconjunto de 100 clases
│   │   └── videos/            # Videos descargados
│   ├── processed_frames/       # Frames procesados (opcional)
│   └── features/               # Features extraídas (.npy)
├── notebooks/
│   └── 01_preprocessing.ipynb  # Notebook para preprocesamiento
├── scripts/
│   ├── preprocess.py           # Script principal de preprocesamiento
│   └── utils.py                # Funciones auxiliares
├── models/
│   ├── arch.py                 # Arquitectura CNN+BiLSTM+Attention
│   └── checkpoints/            # Pesos del modelo entrenado
├── requirements.txt            # Dependencias del proyecto
├── environment.yml             # Entorno conda
└── .gitignore                  # Archivos a ignorar
```

## Instalación

### Opción 1: Usando conda

```bash
# Crear entorno desde el archivo
conda env create -f environment.yml

# Activar entorno
conda activate kophos
```

### Opción 2: Usando pip

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Pipeline de Preprocesamiento

El preprocesamiento consta de los siguientes pasos:

1. **Carga de Metadatos**: Lee `WLASL_v0.3.json` y `nslt_100.json`
2. **Identificación del Bounding Box**: Extrae el bbox [ymin, xmin, ymax, xmax] para cada video
3. **Recorte y Redimensión**: Aplica el bbox y redimensiona a 224×224 píxeles
4. **Muestreo Temporal**: Asegura exactamente 30 frames por video
   - Si tiene más: muestreo uniforme
   - Si tiene menos: repite el último frame (padding)
5. **Extracción de Features**: Usa MobileNetV2 para obtener representaciones de 1280 dimensiones

### Uso del Script de Preprocesamiento

```bash
python scripts/preprocess.py \
    --wlasl_json data/raw/WLASL_v0.3.json \
    --nslt_json data/raw/nslt_100.json \
    --videos_dir data/raw/videos \
    --output_dir data \
    [--no_features]      # No extraer features (solo preprocesar frames)
    [--save_frames]      # Guardar frames procesados (ocupa mucho espacio)
```

### Uso del Notebook

Para un flujo más interactivo, usa el notebook `01_preprocessing.ipynb`:

```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

O súbelo a Google Colab para usar GPU gratis.

## Arquitectura del Modelo

El modelo implementa una arquitectura híbrida:

### Componentes

1. **CNN (MobileNetV2)**
   - Extrae características espaciales de cada frame
   - Output: 1280 features por frame
   - Puede usar pesos pre-entrenados de ImageNet

2. **BiLSTM**
   - Modela dependencias temporales bidireccionales
   - Captura contexto de frames anteriores y posteriores

3. **Attention Layer**
   - Pondera la importancia de cada frame
   - Permite interpretar qué frames son más relevantes

4. **Clasificador**
   - Capas fully connected
   - Output: 100 clases (NSLT-100)

### Uso del Modelo

```python
from models.arch import ASLRecognitionModel, ASLRecognitionModelV2

# Opción 1: Modelo end-to-end (procesa videos directamente)
model = ASLRecognitionModel(
    num_classes=100,
    lstm_hidden_dim=256,
    lstm_num_layers=2,
    dropout=0.5
)

# Opción 2: Modelo con features pre-extraídas (más eficiente)
model = ASLRecognitionModelV2(
    num_classes=100,
    feature_dim=1280,
    lstm_hidden_dim=256
)
```

## Formato de Datos

### Features Guardadas

Cada archivo `.npy` en `data/features/` contiene:

```python
{
    'features': np.array,      # Shape: (30, 1280)
    'video_id': str,           # ID del video
    'action_label': int,       # Índice de clase (0-99)
    'subset': str,             # 'train', 'val', o 'test'
    'gloss': str               # Nombre de la señal
}
```

## Próximos Pasos

1. ✅ Preprocesamiento completado
2. ⬜ Implementar DataLoader para entrenamiento
3. ⬜ Entrenar modelo CNN+BiLSTM+Attention
4. ⬜ Evaluación y métricas
5. ⬜ Integración con app móvil

## Requisitos del Sistema

- Python 3.8+
- PyTorch 1.10+
- GPU recomendada (pero no obligatoria)
- ~10GB de espacio para datos

## Contribución

Este proyecto está en desarrollo activo. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## Licencia

Ver archivo [LICENSE](LICENSE)

## Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

