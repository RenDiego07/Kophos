"""
Script de ejemplo para probar el preprocesamiento con un solo video del subset TEST
"""

import sys
import os
import numpy as np

# Añadir scripts al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from utils import load_wlasl_metadata, load_nslt_subset, get_video_info
from preprocess import VideoPreprocessor


def test_single_video():
    """
    Prueba el preprocesamiento con un video del subset TEST
    """
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    WLASL_JSON = os.path.join(PROJECT_ROOT, 'data/raw/WLASL_v0.3.json')
    NSLT_JSON = os.path.join(PROJECT_ROOT, 'data/raw/nslt_100.json')
    VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'data/raw/videos_nslt_100/test')
    
    print("=" * 60)
    print("TEST PREPROCESAMIENTO - Video del subset TEST")
    print("=" * 60)
    
    # Verificar que existe la carpeta de videos
    if not os.path.exists(VIDEOS_DIR):
        print(f"\nError: No existe la carpeta {VIDEOS_DIR}")
        print("   Ejecuta primero el script organize_nslt100_videos.py")
        return
    
    # Listar videos disponibles en test
    test_videos = [f.replace('.mp4', '') for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
    
    if not test_videos:
        print(f"\nNo hay videos en {VIDEOS_DIR}")
        return
    
    print(f"\nVideos disponibles en test: {len(test_videos)}")
    
    # Seleccionar el primer video
    video_id = test_videos[0]
    print(f"\nVideo seleccionado: {video_id}")
    
    print("\nCargando metadatos...")
    wlasl_data = load_wlasl_metadata(WLASL_JSON)
    nslt_data = load_nslt_subset(NSLT_JSON)
    
    # Obtener información del video
    video_info = get_video_info(video_id, wlasl_data, nslt_data)
    
    print("\nInformacion del video:")
    print(f"  Video ID: {video_id}")
    print(f"  Gloss: {video_info['gloss']}")
    print(f"  Label: {video_info['action_label']}")
    print(f"  Subset: {video_info['subset']}")
    print(f"  Bbox: {video_info['bbox']}")
    
    # Construir path al video
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        print(f"\nVideo no encontrado: {video_path}")
        return
    
    # Preprocesar video
    print("\nPreprocesando video...")
    print(f"  Target size: 224x224")
    print(f"  Target frames: 30")
    
    preprocessor = VideoPreprocessor(target_size=(224, 224), target_frames=30)
    
    try:
        video_frames = preprocessor.preprocess_video(video_path, video_info['bbox'])
        
        print(f"\nPreprocesamiento exitoso!")
        print(f"\nResultado:")
        print(f"  Shape: {video_frames.shape}")
        print(f"  Dtype: {video_frames.dtype}")
        print(f"  Min value: {video_frames.min():.2f}")
        print(f"  Max value: {video_frames.max():.2f}")
        print(f"  Mean value: {video_frames.mean():.2f}")
        print(video_frames)
        
        # Guardar el array (opcional)
        output_path = os.path.join(PROJECT_ROOT, f'test_frames_{video_id}.npy')
        np.save(output_path, video_frames)
        print(f"\nArray guardado en: {output_path}")
        print(f"   Tamano del archivo: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        print("\n" + "=" * 60)
        print("COMPLETADO")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError durante el preprocesamiento: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_video()
