import os
import shutil
import json
import random

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(BASE_DIR, "data", "ASL_34_Classes")
FLAT_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "ASL_34_Flat_Videos")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "asl_34_citizen_index.json")

# 34 clases ordenadas alfabéticamente
CLASS_MAP = {
    "ASK": 0, "BAD": 1, "BECOME": 2, "CHAT": 3, "FIND": 4,
    "FINISH": 5, "GOOD": 6, "HELLO": 7, "HELP": 8, "HOSPITAL": 9,
    "KNOW": 10, "LAUGH": 11, "LIKE": 12, "MAKE": 13, "ME": 14,
    "MORE": 15, "NEED": 16, "NO": 17, "PLAY": 18, "PLEASE": 19,
    "READ": 20, "SHOW": 21, "START": 22, "TELL": 23, "THINK": 24,
    "TO": 25, "UNDERSTAND": 26, "WANT": 27, "WHERE": 28, "WHO": 29,
    "WHY": 30, "WRITE": 31, "YES": 32, "YOU": 33
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