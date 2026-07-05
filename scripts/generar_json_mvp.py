import os
import shutil
import json

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Origen (tus videos curados) y Destino (la carpeta plana para el preprocesador)
SOURCE_DIR = os.path.join(BASE_DIR, "data", "ASL_5_Classes")
FLAT_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "ASL_5_Flat_Videos")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "asl_citizen_5_index.json")

# Nuevo mapeo estricto de 0 a 4
CLASS_MAP = {
    "HOSPITAL": 0,
    "LAUGH": 1,
    "MAKE": 2,
    "ME": 3,
    "NEED": 4
}

def aplanar_y_generar_json():
    os.makedirs(FLAT_VIDEOS_DIR, exist_ok=True)
    dataset_index = {}
    videos_procesados = 0

    print("Escaneando carpetas curadas y generando JSON de 5 clases...")

    for palabra in CLASS_MAP.keys():
        palabra_path = os.path.join(SOURCE_DIR, palabra)
        if not os.path.exists(palabra_path):
            continue

        for subset in ["train", "val", "test"]:
            subset_path = os.path.join(palabra_path, subset)
            if not os.path.exists(subset_path):
                continue

            for archivo in os.listdir(subset_path):
                if archivo.endswith('.mp4'):
                    # 1. Copiar a la carpeta plana
                    ruta_origen = os.path.join(subset_path, archivo)
                    ruta_destino = os.path.join(FLAT_VIDEOS_DIR, archivo)
                    shutil.copy2(ruta_origen, ruta_destino)

                    # 2. Registrar en el JSON con el formato clásico
                    llave_json = archivo.replace('.mp4', '')
                    label_id = CLASS_MAP[palabra]
                    
                    dataset_index[llave_json] = {
                        "subset": subset,
                        "action": [label_id, 1, -1]
                    }
                    videos_procesados += 1

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset_index, f, indent=4)

    print(f"✅ ¡Listo! Se copiaron {videos_procesados} videos a {FLAT_VIDEOS_DIR}")
    print(f"✅ Nuevo JSON guardado en {OUTPUT_JSON}")

if __name__ == "__main__":
    aplanar_y_generar_json()