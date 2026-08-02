import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

def estoy_bien():
    print("hola")

if __name__ == "__main__":


# 1. Cargamos el CSV forzando que todas las columnas sean texto (evita el error de numpy)
    ruta_csv = "./metadata_mp.csv"
    df = pd.read_csv(ruta_csv, dtype=str)

    # 2. Extraemos la palabra directamente del video_id (asegurándonos de que x sea string)
    df['gloss_corregido'] = df['video_id'].apply(lambda x: str(x).split('-')[-1])

    # 3. Reemplazamos la columna antigua con la corregida y eliminamos la temporal
    df['gloss'] = df['gloss_corregido']
    df = df.drop(columns=['gloss_corregido'])

    # Convertimos action_label a número explícitamente solo para poder hacer el filtro del print
    df['action_label'] = pd.to_numeric(df['action_label'])

    # 4. Sobrescribimos el archivo original
    df.to_csv(ruta_csv, index=False)

    print("¡CSV reparado con éxito!")
    # Verificamos que las clases nuevas (ID 5 en adelante) ya no digan UNKNOWN
    print(df[df['action_label'] >= 5].head())