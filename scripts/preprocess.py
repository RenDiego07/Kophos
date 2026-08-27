"""
ASL Citizen Preprocessing Script - MediaPipe Edition (MVP)
"""

import os
import csv
import json
import argparse
from typing import List

import urllib.request
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225  # 33*3 pose + 21*3 left hand + 21*3 right hand

# URLs de modelos para la Tasks API (mediapipe >= 0.10.x y 1.x)
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_POSE_MODEL_PATH = "/tmp/pose_landmarker_full.task"
_HAND_MODEL_PATH = "/tmp/hand_landmarker.task"


def _download_if_missing(url: str, path: str) -> None:
    if not os.path.exists(path):
        print(f"Descargando modelo: {os.path.basename(path)} ...", flush=True)
        urllib.request.urlretrieve(url, path)
        print("  ✅ descargado.", flush=True)


class VideoPreprocessor:
    def __init__(self, target_frames: int = SEQUENCE_LENGTH):
        self.target_frames = target_frames

    def get_sample_indices(self, frame_start: int, frame_end: int, total_frames: int) -> np.ndarray:
        if frame_start < 1:
            frame_start = 1
        if frame_end == -1 or frame_end > total_frames:
            frame_end = int(total_frames * 0.85)

        if frame_start >= frame_end:
            frame_start = 1
            frame_end = total_frames

        indices = np.linspace(frame_start, frame_end - 1, self.target_frames).astype(int)
        return indices

    def preprocess_video(self, video_path: str, frame_start: int, frame_end: int) -> List[np.ndarray]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"OpenCV no pudo abrir: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video sin frames: {video_path}")

        frame_indices = self.get_sample_indices(frame_start, frame_end, total_frames)
        processed_frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()

            if not ret or frame is None:
                processed_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue

            crop_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(crop_rgb)

        cap.release()
        return processed_frames


class MediaPipeExtractor:
    """Extractor basado en la Tasks API — compatible con mediapipe 0.10.x y 1.x."""

    def __init__(self):
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision

        _download_if_missing(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        _download_if_missing(_HAND_MODEL_URL, _HAND_MODEL_PATH)

        pose_opts = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_POSE_MODEL_PATH),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = _mp_vision.PoseLandmarker.create_from_options(pose_opts)

        hand_opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = _mp_vision.HandLandmarker.create_from_options(hand_opts)

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose = np.zeros(33 * 3, dtype=np.float32)
        left_hand = np.zeros(21 * 3, dtype=np.float32)
        right_hand = np.zeros(21 * 3, dtype=np.float32)

        try:
            pose_res = self._pose.detect(mp_img)
            if pose_res.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0]],
                    dtype=np.float32,
                ).flatten()
        except Exception:
            pass

        try:
            hand_res = self._hands.detect(mp_img)
            for i, handedness_list in enumerate(hand_res.handedness):
                label = handedness_list[0].category_name  # "Left" o "Right"
                arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]],
                    dtype=np.float32,
                ).flatten()
                if label == "Left":
                    left_hand = arr
                else:
                    right_hand = arr
        except Exception:
            pass

        return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

    def extract_features(self, video_frames: List[np.ndarray]) -> np.ndarray:
        features = [self.extract_frame_features(frame) for frame in video_frames]
        return np.array(features, dtype=np.float32)

    def close(self):
        self._pose.close()
        self._hands.close()


def process_dataset(nslt_json_path: str, videos_dir: str, output_dir: str, limit: int = None):
    print("Inicializando pipeline para ASL Citizen...", flush=True)

    features_dir = os.path.join(output_dir, "features_mp")
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(features_dir, subset), exist_ok=True)

    metadata_csv_path = os.path.join(output_dir, "metadata_mp.csv")
    
    with open(nslt_json_path, 'r') as f:
        nslt_data = json.load(f)

    preprocessor = VideoPreprocessor(target_frames=SEQUENCE_LENGTH)
    extractor = MediaPipeExtractor()

    processed_count = 0
    missing_count = 0
    metadata_rows = []

    for video_id, info in tqdm(nslt_data.items()):
        if limit and processed_count >= limit:
            break

        subset = info["subset"]
        action_label = info["action"][0]
        frame_start = info["action"][1]
        frame_end = info["action"][2]

        video_path = os.path.join(videos_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            missing_count += 1
            continue

        feature_path = os.path.join(features_dir, subset, f"{video_id}.npy")

        if not os.path.exists(feature_path):
            try:
                processed_frames = preprocessor.preprocess_video(video_path, frame_start, frame_end)
                features = extractor.extract_features(processed_frames)
                np.save(feature_path, features)
            except Exception as e:
                print(f"Error procesando {video_id}: {e}")
                continue

        CLASSES = [
            "BAD", "BECOME", "FIND", "FINISH", "GOOD", "HELLO", "HELP", "HOSPITAL",
            "KNOW", "LAUGH", "LIKE", "MAKE", "ME", "MORE", "NEED", "NO", "PLAY",
            "PLEASE", "READ", "SHOW", "START", "TELL", "THINK", "WANT", "WHERE",
            "WHO", "WHY", "YES", "YOU",
        ]
        reverse_map = {i: cls for i, cls in enumerate(CLASSES)}

        metadata_rows.append({
            "video_id": video_id,
            "subset": subset,
            "action_label": action_label,
            "gloss": reverse_map.get(action_label, "UNKNOWN"),
            "feature_path": feature_path
        })
        
        processed_count += 1

    extractor.close()

    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "subset", "action_label", "gloss", "feature_path"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nProceso completado. Videos procesados: {processed_count}, Faltantes: {missing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ASL Citizen videos")
    parser.add_argument("--nslt_json", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    process_dataset(args.nslt_json, args.videos_dir, args.output_dir, args.limit)