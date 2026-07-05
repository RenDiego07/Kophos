import os
import shutil
import random

# 1. Definir rutas absolutas de forma dinámica
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Origen: donde están tus videos actualmente
SOURCE_DIR = os.path.join(BASE_DIR, "data", "raw", "ASL_Citizen", "videos")

# Destino: la nueva carpeta organizada para tu MVP
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ASL_5_Classes")

# 2. Tu léxico definido
LEXICON = ["MAKE", "LAUGH", "ME", "NEED", "HOSPITAL"]
SUBSETS = ["train", "val", "test"]

def organizar_archivos():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: No se encontró el directorio de origen: {SOURCE_DIR}")
        return

    # Crear la estructura de carpetas vacías
    print("Creando estructura de directorios...")
    for palabra in LEXICON:
        for subset in SUBSETS:
            ruta_carpeta = os.path.join(OUTPUT_DIR, palabra, subset)
            os.makedirs(ruta_carpeta, exist_ok=True)

    # Leer todos los videos disponibles
    archivos = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    
    # Agrupar los videos por palabra
    videos_agrupados = {palabra: [] for palabra in LEXICON}
    
    for archivo in archivos:
        nombre_base = archivo.replace('.mp4', '')
        if '-' in nombre_base:
            palabra = nombre_base.split('-')[-1]
            if palabra in LEXICON:
                videos_agrupados[palabra].append(archivo)

    # 3. Distribuir y copiar los archivos
    print("\nIniciando distribución de archivos...")
    for palabra, lista_videos in videos_agrupados.items():
        total = len(lista_videos)
        if total == 0:
            print(f"Advertencia: No se encontraron videos para '{palabra}'.")
            continue
            
        # Mezclar aleatoriamente para evitar sesgos
        random.shuffle(lista_videos)
        
        # Calcular los límites para los splits (70% / 15% / 15%)
        train_lim = int(total * 0.70)
        val_lim = int(total * 0.85)
        
        # Copiar cada video a su carpeta correspondiente
        for idx, archivo in enumerate(lista_videos):
            if idx < train_lim:
                subset = "train"
            elif idx < val_lim:
                subset = "val"
            else:
                subset = "test"
                
            ruta_origen = os.path.join(SOURCE_DIR, archivo)
            ruta_destino = os.path.join(OUTPUT_DIR, palabra, subset, archivo)
            
            # shutil.copy2 copia el archivo conservando sus metadatos
            shutil.copy2(ruta_origen, ruta_destino)
            
        print(f"✅ {palabra}: {total} videos organizados (Train: {train_lim}, Val: {val_lim - train_lim}, Test: {total - val_lim})")

    print(f"\n¡Éxito! Todos los archivos están listos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    organizar_archivos()