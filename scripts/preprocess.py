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
import numpy as np


def normalize_video_id(video_id: Any) -> str:
    return str(video_id).zfill(5)


def load_wlasl_metadata(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nslt_subset(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        normalize_video_id(video_id): info
        for video_id, info in data.items()
    }


def build_wlasl_video_index(wlasl_data: List[Dict]) -> Dict[str, Dict]:
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


def is_valid_bbox(bbox: Optional[List[int]]) -> bool:
    if bbox is None:
        return False

    if not isinstance(bbox, list):
        return False

    if len(bbox) != 4:
        return False

    x_min, y_min, x_max, y_max = bbox

    return (x_max - x_min) > 0 and (y_max - y_min) > 0


def get_video_info(
    video_id: str,
    wlasl_data: List[Dict],
    nslt_data: Dict
) -> Dict:
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
        "bbox": bbox,
        "fps": video_instance.get("fps"),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "url": video_instance.get("url"),
        "signer_id": video_instance.get("signer_id"),
        "source": video_instance.get("source"),
    }


def find_video_path(
    videos_dir: str,
    video_id: str,
    subset: str
) -> Optional[str]:
    video_id = normalize_video_id(video_id)

    video_path = os.path.join(
        videos_dir,
        subset,
        f"{video_id}.mp4"
    )

    if os.path.exists(video_path):
        return video_path

    return None


def safe_crop(
    frame: np.ndarray,
    bbox: List[int],
    padding: float = 0.15
):
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


def get_class_distribution(nslt_data: Dict) -> Dict:
    distribution = {
        "train": {},
        "val": {},
        "test": {}
    }

    for _, info in nslt_data.items():
        subset = info["subset"]
        action_label = info["action"][0]

        if subset not in distribution:
            distribution[subset] = {}

        if action_label not in distribution[subset]:
            distribution[subset][action_label] = 0

        distribution[subset][action_label] += 1

    return distribution


def print_dataset_stats(nslt_data: Dict) -> None:
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
    video_index = build_wlasl_video_index(wlasl_data)

    nslt_ids = set(normalize_video_id(video_id) for video_id in nslt_data.keys())
    wlasl_ids = set(video_index.keys())

    missing_in_wlasl = sorted(list(nslt_ids - wlasl_ids))
    found = sorted(list(nslt_ids & wlasl_ids))

    return {
        "total_nslt": len(nslt_ids),
        "found_in_wlasl": len(found),
        "missing_in_wlasl": len(missing_in_wlasl),
        "missing_ids": missing_in_wlasl,
    }


def print_metadata_alignment_report(
    wlasl_data: List[Dict],
    nslt_data: Dict
) -> None:
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