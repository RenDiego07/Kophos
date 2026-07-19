import os
import shutil
import json
import random

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(BASE_DIR, "data", "ASL_15_Classes") # Ajusta el nombre si es necesario
FLAT_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "ASL_15_Flat_Videos")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "asl_15_citizen_index.json")

# Nuevo mapeo estricto con las 16 clases de tu árbol
CLASS_MAP = {
    "HOSPITAL": 0, "LAUGH": 1, "MAKE": 2, "ME": 3, "NEED": 4,
    "READ": 5, "SHOW": 6, "START": 7, "STOP": 8, "TELL": 9,
    "THINK": 10, "TO": 11, "UNDERSTAND": 12, "WAIT": 13, "WANT": 14,
    "WRITE": 15
}

def aplanar_y_generar_json():
    os.makedirs(FLAT_VIDEOS_DIR, exist_ok=True)
    dataset_index = {}
    videos_procesados = 0

    print(f"Escaneando {len(CLASS_MAP)} clases y generando JSON...")

    for palabra, label_id in CLASS_MAP.items():
        palabra_path = os.path.join(SOURCE_DIR, palabra)
        if not os.path.exists(palabra_path):
            print(f"⚠️ Carpeta no encontrada: {palabra}")
            continue

        # Comprobamos si la carpeta ya tiene la estructura train/val/test
        tiene_subsets = any(os.path.exists(os.path.join(palabra_path, s)) for s in ["train", "val", "test"])

        if tiene_subsets:
            # LÓGICA ORIGINAL: Para carpetas ya estructuradas (HOSPITAL, ME, etc.)
            for subset in ["train", "val", "test"]:
                subset_path = os.path.join(palabra_path, subset)
                if not os.path.exists(subset_path):
                    continue

                for archivo in os.listdir(subset_path):
                    if archivo.endswith('.mp4'):
                        procesar_video(archivo, subset_path, subset, label_id, dataset_index)
                        videos_procesados += 1
        else:
            # LÓGICA NUEVA: Para carpetas planas (READ, SHOW, etc.)
            videos = [f for f in os.listdir(palabra_path) if f.endswith('.mp4')]
            
            # Mezclamos aleatoriamente para evitar sesgos
            random.seed(42) # Semilla fija para reproducibilidad
            random.shuffle(videos)
            
            # Calculamos índices para split 80/10/10
            total = len(videos)
            train_cut = int(0.8 * total)
            val_cut = int(0.9 * total)

            for i, archivo in enumerate(videos):
                # Asignamos el subset dinámicamente
                if i < train_cut:
                    subset = "train"
                elif i < val_cut:
                    subset = "val"
                else:
                    subset = "test"

                procesar_video(archivo, palabra_path, subset, label_id, dataset_index)
                videos_procesados += 1

    # Guardar el JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset_index, f, indent=4)

    print(f"✅ ¡Listo! Se procesaron {videos_procesados} videos hacia {FLAT_VIDEOS_DIR}")
    print(f"✅ Nuevo JSON guardado en {OUTPUT_JSON}")


def procesar_video(archivo, origen_path, subset, label_id, dataset_index):
    """Función auxiliar para no repetir código de copiado y registro."""
    ruta_origen = os.path.join(origen_path, archivo)
    ruta_destino = os.path.join(FLAT_VIDEOS_DIR, archivo)
    
    # 1. Copiar a la carpeta plana
    shutil.copy2(ruta_origen, ruta_destino)

    # 2. Registrar en el diccionario JSON
    llave_json = archivo.replace('.mp4', '')
    dataset_index[llave_json] = {
        "subset": subset,
        "action": [label_id, 1, -1]
    }

if __name__ == "__main__":
    aplanar_y_generar_json()