import os
import json
import random

# 1. Rutas de tu entorno local
VIDEO_DIR = "data/raw/ASL_Citizen/videos"
OUTPUT_JSON = "data/asl_citizen_13_index.json"

# 2. Tu propio diccionario de clases (0 a 12)
# Nota: Mapeamos tanto "TRIP" como "HURDLE-TRIP" al ID 12
CLASS_MAP = {
    "BREAKFAST": 0,
    "COOL": 1,
    "DEAF": 2,
    "EAT": 3,
    "HOSPITAL": 4,
    "LAUGH": 5,
    "MAKE": 6,
    "ME": 7,
    "MOVIE": 8,
    "NEED": 9,
    "NIGHT": 10,
    "PARTY": 11,
    "TRIP": 12,
    "HURDLE-TRIP": 12
}

def generar_indice():
    # Asegurarnos de que el directorio existe
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: No se encuentra la ruta {VIDEO_DIR}")
        return

    archivos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"Se encontraron {len(archivos)} videos para procesar.")

    # Agrupar videos por palabra para hacer un split balanceado
    videos_por_clase = {v: [] for v in CLASS_MAP.values()}
    
    # Estructura final del JSON
    dataset_index = {}

    # 3. Extraer etiquetas y asignar IDs
    for archivo in archivos:
        # Extraer el ID del video y la palabra (ej: "006918970578539518-PARTY.mp4")
        nombre_base = archivo.replace('.mp4', '')
        
        if '-' in nombre_base:
            # Separamos por guion (tomamos todo lo que está después del primer guion por si hay nombres como HURDLE-TRIP)
            partes = nombre_base.split('-')
            video_id = partes[0]
            palabra_cruda = '-'.join(partes[1:]) 
            
            if palabra_cruda in CLASS_MAP:
                label_id = CLASS_MAP[palabra_cruda]
                videos_por_clase[label_id].append(archivo)
            else:
                print(f"Advertencia: La palabra '{palabra_cruda}' no está en el diccionario. Se omitirá.")

    # 4. Hacer el split (Train: 70%, Val: 15%, Test: 15%) y construir el JSON
    for label_id, lista_videos in videos_por_clase.items():
        random.shuffle(lista_videos)
        total = len(lista_videos)
        
        train_lim = int(total * 0.70)
        val_lim = int(total * 0.85)
        
        for idx, archivo in enumerate(lista_videos):
            # Determinar a qué subset pertenece
            if idx < train_lim:
                subset = "train"
            elif idx < val_lim:
                subset = "val"
            else:
                subset = "test"
                
            # Nombre de la llave sin la extensión para que el DataLoader lo empate fácil
            llave_json = archivo.replace('.mp4', '')
            
            # Formato exacto que espera tu archivo utils.py
            # [label_id, frame_start, frame_end] (-1 indica leer hasta el final)
            dataset_index[llave_json] = {
                "subset": subset,
                "action": [label_id, 1, -1]
            }

    # 5. Guardar el archivo JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset_index, f, indent=4)

    print(f"¡Índice generado con éxito en {OUTPUT_JSON}!")
    print(f"Total de videos indexados: {len(dataset_index)}")

if __name__ == "__main__":
    generar_indice()