# Kophos — Dataset Pipeline Workflow

Pipeline completo para agregar nuevas clases al dataset de entrenamiento ASL.

---

## Estructura de directorios clave

```
data/
├── ASL_14_new_classes/          # Ejemplo: carpeta plana de videos nuevos (ID-WORD.mp4)
├── ASL_34_Classes/              # Árbol organizado: WORD/train|val|test/*.mp4
├── ASL_34_Flat_Videos/          # Todos los videos en una sola carpeta (para DataLoader)
└── asl_34_citizen_index.json    # Índice final para entrenamiento

scripts/
├── separar_carpetas.py          # Paso 1 — organiza videos planos en árbol train/val/test
├── generar_json_mvp.py          # Paso 2 — genera JSON índice + carpeta plana
└── flip_augment.py              # Paso 3 — augmentación horizontal + actualiza JSON
```

---

## Paso 1 — Organizar videos nuevos

**Script:** `scripts/separar_carpetas.py`

**Cuándo usarlo:** cuando se agrega una carpeta plana de videos nuevos con formato `ID-WORD.mp4`.

**Parámetros a editar antes de correr:**

```python
SOURCE_DIR  = "data/<carpeta_plana_nueva>"   # origen: videos sin estructura
OUTPUT_DIR  = "data/ASL_<N>_Classes"          # destino: árbol con clases existentes
LEXICON     = ["WORD1", "WORD2", ...]          # solo las palabras NUEVAS a agregar
```

**Resultado:** crea `OUTPUT_DIR/WORD/train|val|test/` con split 70/15/15.

```bash
python scripts/separar_carpetas.py
```

> **Nota:** Las clases ya existentes en `OUTPUT_DIR` no se tocan. El script solo agrega las nuevas.

---

## Paso 2 — Generar índice JSON y carpeta plana

**Script:** `scripts/generar_json_mvp.py`

**Cuándo usarlo:** siempre después del Paso 1 (o cuando se quiere regenerar el índice desde cero).

**Parámetros a editar:**

```python
SOURCE_DIR   = "data/ASL_<N>_Classes"          # árbol de clases
FLAT_VIDEOS_DIR = "data/ASL_<N>_Flat_Videos"   # destino plano para DataLoader
OUTPUT_JSON  = "data/asl_<n>_citizen_index.json"

CLASS_MAP = {
    "WORD1": 0,
    "WORD2": 1,
    # ... todas las clases en orden alfabético, índice 0..N-1
}
```

**Maneja automáticamente:**
- Carpetas con estructura `train/val/test` (clases nuevas)
- Carpetas planas sin subsets (clases antiguas con o sin `_flip`)

**Resultado:** JSON con entradas `"ID-WORD": {"subset": "train|val|test", "action": [label, 1, -1]}` y copia de todos los videos a `FLAT_VIDEOS_DIR`.

```bash
python scripts/generar_json_mvp.py
```

---

## Paso 3 — Flip augmentation

**Script:** `scripts/flip_augment.py`

**Cuándo usarlo:** después del Paso 1 y antes o después del Paso 2. Genera versiones espejadas horizontalmente de cada video original.

**Parámetros a editar:**

```python
CLASSES = ["WORD1", "WORD2", ...]   # todas las clases (34 en la iteración actual)
# CLASS_INDEX se genera automáticamente desde CLASSES (mismo orden que CLASS_MAP)

CLASSES_DIR = "data/ASL_<N>_Classes"
FLAT_DIR    = "data/ASL_<N>_Flat_Videos"
INDEX_PATH  = "data/asl_<n>_citizen_index.json"
```

**Comportamiento:**
- Omite archivos que ya contienen `_flip` en el nombre → no duplica augmentaciones.
- Guarda el video volteado junto al original (misma subcarpeta).
- Copia el video volteado a `FLAT_DIR`.
- Agrega las nuevas entradas al JSON con el mismo label que el video original.

```bash
python scripts/flip_augment.py
```

> **Importante:** Si el JSON fue regenerado desde cero en el Paso 2 (sobrescribe entradas previas), correr el Paso 3 **después** del Paso 2. El script actualiza el JSON existente sin borrarlo.

---

## Orden recomendado para una nueva iteración

```
1. Colocar videos nuevos en data/ASL_<X>_new_classes/  (formato ID-WORD.mp4)

2. Editar separar_carpetas.py → SOURCE_DIR, OUTPUT_DIR, LEXICON
   python scripts/separar_carpetas.py

3. Editar flip_augment.py → CLASSES, CLASSES_DIR, FLAT_DIR, INDEX_PATH
   python scripts/flip_augment.py

4. Editar generar_json_mvp.py → SOURCE_DIR, FLAT_VIDEOS_DIR, OUTPUT_JSON, CLASS_MAP
   python scripts/generar_json_mvp.py
```

> El Paso 4 siempre va al final porque regenera el JSON desde el estado real del filesystem, capturando tanto los originales como los `_flip` generados en el Paso 3.

---

## Verificación rápida post-pipeline

```python
import json
from collections import Counter

with open("data/asl_<n>_citizen_index.json") as f:
    data = json.load(f)

labels  = Counter(v["action"][0] for v in data.values())
subsets = Counter(v["subset"] for v in data.values())
flips   = sum(1 for k in data if "_flip" in k)

print(f"Total: {len(data)} | Originals: {len(data)-flips} | Flips: {flips}")
print(f"Subsets: {dict(subsets)}")
for label, count in sorted(labels.items()):
    print(f"  {label}: {count} videos")
```

**Señales de alerta:**
- Algún label no aparece → la clase no tiene videos en su carpeta (revisar Paso 1).
- `Flips == 0` → el Paso 3 no corrió o corrió antes de que existieran videos.
- `FLAT_VIDEOS_DIR` vacío → el Paso 4 no pudo copiar archivos; verificar permisos y volver a correr.

---

## Iteración actual

| Versión | Clases | Originals | Flips | Total | JSON |
|---------|--------|-----------|-------|-------|------|
| 34      | 34     | 1,012     | 1,603 | 2,615 | `asl_34_citizen_index.json` |
