"""
Utilidades para cargar y manipular los metadatos de WLASL y NSLT-100
"""
import json
import os
from typing import Dict, List, Tuple


def load_wlasl_metadata(json_path: str) -> List[Dict]:
    """
    Carga el archivo WLASL_v0.3.json
    
    Args:
        json_path: Ruta al archivo WLASL_v0.3.json
    
    Returns:
        Lista de diccionarios con metadatos de cada gloss
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_nslt_subset(json_path: str) -> Dict:
    """
    Carga el archivo nslt_100.json (subconjunto de videos)
    
    Args:
        json_path: Ruta al archivo nslt_100.json
    
    Returns:
        Diccionario con video_id como key y metadata como valor
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_video_info(video_id: str, wlasl_data: List[Dict], nslt_data: Dict) -> Dict:
    """
    Obtiene información completa de un video específico
    
    Args:
        video_id: ID del video
        wlasl_data: Datos de WLASL completos
        nslt_data: Datos del subconjunto NSLT-100
    
    Returns:
        Diccionario con información completa del video
    """
    if video_id not in nslt_data:
        raise ValueError(f"Video ID {video_id} no encontrado en NSLT-100")
    
    # Obtener info del subconjunto
    nslt_info = nslt_data[video_id]
    action_label = nslt_info['action'][0]  # Índice de la clase
    subset = nslt_info['subset']
    
    # Buscar en WLASL para obtener bbox y otros detalles
    video_instance = None
    gloss_name = None
    
    for gloss_entry in wlasl_data:
        for instance in gloss_entry['instances']:
            if instance['video_id'] == video_id:
                video_instance = instance
                gloss_name = gloss_entry['gloss']
                break
        if video_instance:
            break
    
    if not video_instance:
        raise ValueError(f"Video ID {video_id} no encontrado en WLASL")
    
    return {
        'video_id': video_id,
        'gloss': gloss_name,
        'action_label': action_label,
        'subset': subset,
        'bbox': video_instance['bbox'],  # [ymin, xmin, ymax, xmax]
        'fps': video_instance['fps'],
        'frame_start': video_instance['frame_start'],
        'frame_end': video_instance['frame_end'],
        'url': video_instance['url'],
        'signer_id': video_instance.get('signer_id'),
        'source': video_instance.get('source')
    }


def get_largest_bbox(bboxes: List[List[int]]) -> List[int]:
    """
    Identifica el bounding box más grande entre varias instancias
    
    Args:
        bboxes: Lista de bounding boxes [ymin, xmin, ymax, xmax]
    
    Returns:
        El bbox más grande
    """
    if not bboxes:
        raise ValueError("Lista de bboxes vacía")
    
    if len(bboxes) == 1:
        return bboxes[0]
    
    max_area = 0
    largest_bbox = bboxes[0]
    
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = bbox
        area = (ymax - ymin) * (xmax - xmin)
        if area > max_area:
            max_area = area
            largest_bbox = bbox
    
    return largest_bbox


def get_class_distribution(nslt_data: Dict) -> Dict:
    """
    Calcula la distribución de clases en el dataset
    
    Args:
        nslt_data: Datos del subconjunto NSLT-100
    
    Returns:
        Diccionario con la distribución de clases por subset
    """
    distribution = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for video_id, info in nslt_data.items():
        subset = info['subset']
        action = info['action'][0]
        
        if action not in distribution[subset]:
            distribution[subset][action] = 0
        distribution[subset][action] += 1
    
    return distribution


def print_dataset_stats(nslt_data: Dict):
    """
    Imprime estadísticas del dataset
    
    Args:
        nslt_data: Datos del subconjunto NSLT-100
    """
    dist = get_class_distribution(nslt_data)
    
    print("=" * 60)
    print("ESTADÍSTICAS DEL DATASET NSLT-100")
    print("=" * 60)
    
    for subset in ['train', 'val', 'test']:
        total_videos = sum(dist[subset].values())
        num_classes = len(dist[subset])
        print(f"\n{subset.upper()}:")
        print(f"  Total de videos: {total_videos}")
        print(f"  Número de clases: {num_classes}")
        if total_videos > 0:
            print(f"  Videos por clase (promedio): {total_videos / num_classes:.2f}")
    
    print("\n" + "=" * 60)
