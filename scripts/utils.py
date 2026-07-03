"""
Utilidades para cargar y manipular los metadatos de WLASL y NSLT-100.

Notas importantes:
- En WLASL, bbox viene en formato [x_min, y_min, x_max, y_max].
- En NSLT, action viene en formato [class_id, frame_start, frame_end].
- Los video_id deben mantenerse como string con ceros iniciales.
"""

import json
import os
from typing import Dict, List, Optional, Any


def normalize_video_id(video_id: Any) -> str:
    """
    Normaliza el video_id para conservar ceros iniciales.

    Ejemplo:
        5237 -> "05237"
        "05237" -> "05237"

    Args:
        video_id: ID del video como str o int.

    Returns:
        video_id como string de 5 dígitos.
    """
    return str(video_id).zfill(5)


def load_wlasl_metadata(json_path: str) -> List[Dict]:
    """
    Carga el archivo WLASL_v0.3.json.

    Args:
        json_path: Ruta al archivo WLASL_v0.3.json.

    Returns:
        Lista de diccionarios con metadatos de cada gloss.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_nslt_subset(json_path: str) -> Dict:
    """
    Carga el archivo nslt_100.json.

    Args:
        json_path: Ruta al archivo nslt_100.json.

    Returns:
        Diccionario con video_id como key y metadata como valor.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalizar keys para evitar problemas con ceros iniciales
    normalized_data = {
        normalize_video_id(video_id): info
        for video_id, info in data.items()
    }

    return normalized_data


def build_wlasl_video_index(wlasl_data: List[Dict]) -> Dict[str, Dict]:
    """
    Construye un índice para buscar videos de WLASL más rápido.

    Args:
        wlasl_data: Datos completos de WLASL.

    Returns:
        Diccionario con video_id como key y metadata como valor.
    """
    video_index = {}

    for gloss_entry in wlasl_data:
        gloss_name = gloss_entry.get("gloss")

        for instance in gloss_entry.get("instances", []):
            video_id = normalize_video_id(instance.get("video_id"))

            video_index[video_id] = {
                "gloss": gloss_name,
                "instance": instance
            }

    return video_index


def get_video_info(
    video_id: str,
    wlasl_data: List[Dict],
    nslt_data: Dict
) -> Dict:
    """
    Obtiene información completa de un video específico.

    Importante:
    - bbox se toma de WLASL.
    - action_label, frame_start, frame_end y subset se toman de NSLT.
    - bbox está en formato [x_min, y_min, x_max, y_max].

    Args:
        video_id: ID del video.
        wlasl_data: Datos de WLASL completos.
        nslt_data: Datos del subconjunto NSLT-100.

    Returns:
        Diccionario con información completa del video.
    """
    video_id = normalize_video_id(video_id)

    if video_id not in nslt_data:
        raise ValueError(f"Video ID {video_id} no encontrado en NSLT-100")

    nslt_info = nslt_data[video_id]

    if "action" not in nslt_info or len(nslt_info["action"]) < 3:
        raise ValueError(f"Formato inválido de action para video ID {video_id}")

    action_label = nslt_info["action"][0]
    frame_start = nslt_info["action"][1]
    frame_end = nslt_info["action"][2]
    subset = nslt_info["subset"]

    video_index = build_wlasl_video_index(wlasl_data)

    if video_id not in video_index:
        raise ValueError(f"Video ID {video_id} no encontrado en WLASL")

    video_instance = video_index[video_id]["instance"]
    gloss_name = video_index[video_id]["gloss"]

    bbox = video_instance.get("bbox")

    if not is_valid_bbox(bbox):
        raise ValueError(f"BBox inválido para video ID {video_id}: {bbox}")

    return {
        "video_id": video_id,
        "gloss": gloss_name,
        "action_label": action_label,
        "subset": subset,
        "bbox": bbox,  # [x_min, y_min, x_max, y_max]
        "fps": video_instance.get("fps"),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "url": video_instance.get("url"),
        "signer_id": video_instance.get("signer_id"),
        "source": video_instance.get("source"),
    }


def is_valid_bbox(bbox: Optional[List[int]]) -> bool:
    """
    Valida si un bbox tiene formato correcto.

    Args:
        bbox: Bounding box en formato [x_min, y_min, x_max, y_max].

    Returns:
        True si el bbox es válido, False si no.
    """
    if bbox is None:
        return False

    if not isinstance(bbox, list):
        return False

    if len(bbox) != 4:
        return False

    try:
        x_min, y_min, x_max, y_max = bbox
    except ValueError:
        return False

    width = x_max - x_min
    height = y_max - y_min

    return width > 0 and height > 0


def get_largest_bbox(bboxes: List[List[int]]) -> List[int]:
    """
    Identifica el bounding box más grande entre varias instancias.

    Args:
        bboxes: Lista de bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
        El bbox más grande.
    """
    if not bboxes:
        raise ValueError("Lista de bboxes vacía")

    valid_bboxes = [bbox for bbox in bboxes if is_valid_bbox(bbox)]

    if not valid_bboxes:
        raise ValueError("No hay bboxes válidos en la lista")

    if len(valid_bboxes) == 1:
        return valid_bboxes[0]

    max_area = 0
    largest_bbox = valid_bboxes[0]

    for bbox in valid_bboxes:
        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        if area > max_area:
            max_area = area
            largest_bbox = bbox

    return largest_bbox


def find_video_path(
    videos_dir: str,
    video_id: str,
    subset: Optional[str] = None
) -> Optional[str]:
    """
    Busca el path real de un video.

    Soporta varias estructuras posibles:

    videos_dir/05237.mp4
    videos_dir/train/05237.mp4
    videos_dir/val/05237.mp4
    videos_dir/test/05237.mp4
    videos_dir/videos/05237.mp4
    videos_dir/videos/train/05237.mp4

    Args:
        videos_dir: Carpeta raíz donde están los videos.
        video_id: ID del video.
        subset: train, val o test.

    Returns:
        Ruta al video si existe, None si no existe.
    """
    video_id = normalize_video_id(video_id)

    candidates = [
        os.path.join(videos_dir, f"{video_id}.mp4"),
        os.path.join(videos_dir, "videos", f"{video_id}.mp4"),
    ]

    if subset is not None:
        candidates.extend([
            os.path.join(videos_dir, subset, f"{video_id}.mp4"),
            os.path.join(videos_dir, "videos", subset, f"{video_id}.mp4"),
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def get_class_distribution(nslt_data: Dict) -> Dict:
    """
    Calcula la distribución de clases en el dataset.

    Args:
        nslt_data: Datos del subconjunto NSLT-100.

    Returns:
        Diccionario con la distribución de clases por subset.
    """
    distribution = {
        "train": {},
        "val": {},
        "test": {}
    }

    for video_id, info in nslt_data.items():
        subset = info["subset"]
        action_label = info["action"][0]

        if subset not in distribution:
            distribution[subset] = {}

        if action_label not in distribution[subset]:
            distribution[subset][action_label] = 0

        distribution[subset][action_label] += 1

    return distribution


def print_dataset_stats(nslt_data: Dict) -> None:
    """
    Imprime estadísticas del dataset.

    Args:
        nslt_data: Datos del subconjunto NSLT-100.
    """
    dist = get_class_distribution(nslt_data)

    print("=" * 60)
    print("ESTADÍSTICAS DEL DATASET NSLT-100")
    print("=" * 60)

    total_dataset_videos = 0

    for subset in ["train", "val", "test"]:
        total_videos = sum(dist.get(subset, {}).values())
        num_classes = len(dist.get(subset, {}))
        total_dataset_videos += total_videos

        print(f"\n{subset.upper()}:")
        print(f"  Total de videos: {total_videos}")
        print(f"  Número de clases: {num_classes}")

        if total_videos > 0 and num_classes > 0:
            print(f"  Videos por clase promedio: {total_videos / num_classes:.2f}")

    print(f"\nTOTAL DATASET: {total_dataset_videos} videos")
    print("=" * 60)


def validate_metadata_alignment(
    wlasl_data: List[Dict],
    nslt_data: Dict
) -> Dict:
    """
    Valida que los video_id de NSLT existan dentro de WLASL.

    Args:
        wlasl_data: Datos completos de WLASL.
        nslt_data: Datos del subconjunto NSLT.

    Returns:
        Diccionario con resumen de validación.
    """
    video_index = build_wlasl_video_index(wlasl_data)

    nslt_ids = set(normalize_video_id(video_id) for video_id in nslt_data.keys())
    wlasl_ids = set(video_index.keys())

    missing_in_wlasl = sorted(list(nslt_ids - wlasl_ids))
    found = sorted(list(nslt_ids & wlasl_ids))

    result = {
        "total_nslt": len(nslt_ids),
        "found_in_wlasl": len(found),
        "missing_in_wlasl": len(missing_in_wlasl),
        "missing_ids": missing_in_wlasl,
    }

    return result


def print_metadata_alignment_report(
    wlasl_data: List[Dict],
    nslt_data: Dict
) -> None:
    """
    Imprime un reporte simple de alineación entre NSLT y WLASL.

    Args:
        wlasl_data: Datos completos de WLASL.
        nslt_data: Datos del subconjunto NSLT.
    """
    report = validate_metadata_alignment(wlasl_data, nslt_data)

    print("=" * 60)
    print("VALIDACIÓN NSLT-100 vs WLASL")
    print("=" * 60)
    print(f"Total videos en NSLT: {report['total_nslt']}")
    print(f"Encontrados en WLASL: {report['found_in_wlasl']}")
    print(f"No encontrados en WLASL: {report['missing_in_wlasl']}")

    if report["missing_ids"]:
        print("\nPrimeros IDs faltantes:")
        print(report["missing_ids"][:20])

    print("=" * 60)


def create_metadata_list(
    wlasl_data: List[Dict],
    nslt_data: Dict,
    videos_dir: Optional[str] = None
) -> List[Dict]:
    """
    Crea una lista de metadatos lista para el preprocesamiento.

    Args:
        wlasl_data: Datos completos de WLASL.
        nslt_data: Datos del subconjunto NSLT.
        videos_dir: Carpeta raíz de videos. Si se pasa, agrega video_path.

    Returns:
        Lista de diccionarios con metadata por video.
    """
    metadata = []

    for video_id in nslt_data.keys():
        try:
            info = get_video_info(video_id, wlasl_data, nslt_data)

            if videos_dir is not None:
                info["video_path"] = find_video_path(
                    videos_dir=videos_dir,
                    video_id=info["video_id"],
                    subset=info["subset"]
                )

            metadata.append(info)

        except Exception as e:
            print(f"[ERROR METADATA] video_id={video_id}: {e}")

    return metadata






# ============================================================
# DEBUG / VIDEO PATH VALIDATION
# ============================================================

import cv2
import numpy as np


def find_video_path(
    videos_dir: str,
    video_id: str,
    subset: str
) -> Optional[str]:
    """
    Busca el path real de un video dentro de la estructura:

    videos_nslt_100/
    ├── train/
    ├── val/
    └── test/

    Ejemplo:
        videos_nslt_100/train/05237.mp4

    Args:
        videos_dir: Carpeta raíz donde están train, val y test.
        video_id: ID del video.
        subset: train, val o test.

    Returns:
        Ruta al video si existe. None si no existe.
    """
    video_id = normalize_video_id(video_id)

    video_path = os.path.join(
        videos_dir,
        subset,
        f"{video_id}.mp4"
    )

    if os.path.exists(video_path):
        return video_path

    return None


def debug_video_paths(
    nslt_data: Dict,
    videos_dir: str,
    num_videos: int = 10
) -> None:
    """
    Verifica si los videos existen físicamente en train/val/test.

    Args:
        nslt_data: Diccionario cargado desde nslt_100.json.
        videos_dir: Carpeta raíz videos_nslt_100.
        num_videos: Número de videos a revisar.
    """
    print("=" * 80)
    print("DEBUG VIDEO PATHS")
    print("=" * 80)

    checked = 0
    found = 0
    missing = 0

    for video_id, info in list(nslt_data.items())[:num_videos]:
        video_id = normalize_video_id(video_id)
        subset = info["subset"]

        video_path = find_video_path(
            videos_dir=videos_dir,
            video_id=video_id,
            subset=subset
        )

        exists = video_path is not None

        print(f"\nvideo_id: {video_id}")
        print(f"subset: {subset}")
        print(f"path: {video_path}")
        print(f"exists: {exists}")

        checked += 1

        if exists:
            found += 1
        else:
            missing += 1

    print("\n" + "=" * 80)
    print("RESUMEN PATHS")
    print("=" * 80)
    print(f"Revisados: {checked}")
    print(f"Encontrados: {found}")
    print(f"Faltantes: {missing}")
    print("=" * 80)


def safe_crop(
    frame: np.ndarray,
    bbox: List[int],
    padding: float = 0.15
):
    """
    Recorta el frame usando bbox en formato WLASL:

    bbox = [x_min, y_min, x_max, y_max]

    Si el bbox es inválido, retorna el frame completo como fallback.

    Args:
        frame: Frame leído con OpenCV.
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        padding: Padding proporcional alrededor del bbox.

    Returns:
        crop: Imagen recortada o frame completo.
        status: Estado del crop.
    """
    h, w = frame.shape[:2]

    if bbox is None or len(bbox) != 4:
        return frame, "invalid_bbox_format"

    x_min, y_min, x_max, y_max = bbox

    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width <= 0 or box_height <= 0:
        return frame, "invalid_bbox_size"

    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)

    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(w, int(x_max + pad_x))
    y_max = min(h, int(y_max + pad_y))

    if x_max <= x_min or y_max <= y_min:
        return frame, "invalid_bbox_after_clamp"

    crop = frame[y_min:y_max, x_min:x_max]

    if crop is None or crop.size == 0:
        return frame, "empty_crop"

    return crop, "ok"


def debug_video_extraction(
    wlasl_data: List[Dict],
    nslt_data: Dict,
    videos_dir: str,
    num_videos: int = 5,
    num_frames_per_video: int = 3
) -> None:
    """
    Debuggea la extracción básica antes de correr todo el pipeline.

    Revisa:
    - video_path
    - bbox
    - frame.shape
    - crop.shape
    - rango de frames
    - si OpenCV puede abrir el video

    Args:
        wlasl_data: Metadata de WLASL.
        nslt_data: Metadata de NSLT-100.
        videos_dir: Carpeta raíz videos_nslt_100.
        num_videos: Número de videos a probar.
        num_frames_per_video: Número de frames a revisar por video.
    """
    print("=" * 80)
    print("DEBUG VIDEO EXTRACTION")
    print("=" * 80)

    video_index = build_wlasl_video_index(wlasl_data)

    debug_ids = list(nslt_data.keys())[:num_videos]

    for video_id in debug_ids:
        video_id = normalize_video_id(video_id)

        print("\n" + "=" * 80)
        print(f"VIDEO ID: {video_id}")

        if video_id not in nslt_data:
            print("[ERROR] Video no encontrado en NSLT")
            continue

        nslt_info = nslt_data[video_id]

        subset = nslt_info["subset"]
        action = nslt_info["action"]

        action_label = action[0]
        frame_start = action[1]
        frame_end = action[2]

        print(f"subset: {subset}")
        print(f"action_label: {action_label}")
        print(f"frame_start: {frame_start}")
        print(f"frame_end: {frame_end}")

        if video_id not in video_index:
            print("[ERROR] Video no encontrado en WLASL")
            continue

        wlasl_instance = video_index[video_id]["instance"]
        gloss = video_index[video_id]["gloss"]
        bbox = wlasl_instance["bbox"]

        print(f"gloss: {gloss}")
        print(f"bbox: {bbox}  # [x_min, y_min, x_max, y_max]")

        video_path = find_video_path(
            videos_dir=videos_dir,
            video_id=video_id,
            subset=subset
        )

        print(f"video_path: {video_path}")
        print(f"exists: {video_path is not None}")

        if video_path is None:
            print("[ERROR] No se encontró el video en train/val/test")
            continue

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("[ERROR] OpenCV no pudo abrir el video")
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"total_frames: {total_frames}")
        print(f"fps: {fps}")
        print(f"width: {width}")
        print(f"height: {height}")

        if frame_start < 1:
            frame_start = 1

        if frame_end == -1 or frame_end > total_frames:
            frame_end = total_frames

        if frame_start >= frame_end:
            print("[ERROR] Rango de frames inválido después de ajustar")
            cap.release()
            continue

        test_frame_indices = np.linspace(
            frame_start,
            frame_end - 1,
            num_frames_per_video
        ).astype(int)

        print(f"test_frame_indices: {test_frame_indices}")

        for frame_idx in test_frame_indices:
            print("\n--- FRAME DEBUG ---")
            print(f"frame_idx: {frame_idx}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)

            ret, frame = cap.read()

            print(f"ret: {ret}")

            if not ret or frame is None:
                print("[ERROR] No se pudo leer el frame")
                continue

            print(f"frame.shape: {frame.shape}")

            crop, crop_status = safe_crop(frame, bbox)

            print(f"crop_status: {crop_status}")
            print(f"crop.shape: {crop.shape}")

            if crop.shape[0] <= 0 or crop.shape[1] <= 0:
                print("[ERROR] Crop inválido")
            else:
                print("[OK] Crop válido")

        cap.release()

    print("\n" + "=" * 80)
    print("DEBUG FINALIZADO")
    print("=" * 80)




