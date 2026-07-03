"""
ASL Video Preprocessing Script - MediaPipe Edition

Pipeline:
1. Load WLASL and NSLT-100 metadata.
2. Locate videos inside videos_nslt_100/train, val, test.
3. Use NSLT frame_start/frame_end to sample exactly 30 frames.
4. Crop signer using WLASL bbox format: [x_min, y_min, x_max, y_max].
5. Extract spatial features with MediaPipe Holistic:
   - Pose: 33 landmarks * 3 = 99
   - Left hand: 21 landmarks * 3 = 63
   - Right hand: 21 landmarks * 3 = 63
   - Total: 225 features per frame
6. Save each video as .npy tensor with shape (30, 225).
7. Save metadata CSV for PyTorch Dataset usage.
"""

import os
import csv
import argparse
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

from utils import (
    load_wlasl_metadata,
    load_nslt_subset,
    get_video_info,
    normalize_video_id,
    find_video_path,
    safe_crop,
)


SEQUENCE_LENGTH = 30
FEATURE_DIM = 225


class VideoPreprocessor:
    """
    Preprocess ASL videos:
    1. Open video with OpenCV.
    2. Sample exactly target_frames from frame_start to frame_end.
    3. Crop each frame using bbox.
    4. Convert frame from BGR to RGB for MediaPipe.
    """

    def __init__(self, target_frames: int = SEQUENCE_LENGTH):
        self.target_frames = target_frames

    def get_sample_indices(
        self,
        frame_start: int,
        frame_end: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Generate exactly target_frames frame indices from the useful signing segment.

        NSLT uses frame_start/frame_end. Usually frame_start starts at 1.
        OpenCV uses zero-based indexing, so we later subtract 1 when reading frames.
        """

        if frame_start < 1:
            frame_start = 1

        if frame_end == -1 or frame_end > total_frames:
            frame_end = total_frames

        if frame_start >= frame_end:
            raise ValueError(
                f"Invalid frame range: frame_start={frame_start}, "
                f"frame_end={frame_end}, total_frames={total_frames}"
            )

        indices = np.linspace(
            frame_start,
            frame_end - 1,
            self.target_frames
        ).astype(int)

        return indices

    def preprocess_video(
        self,
        video_path: str,
        bbox: List[int],
        frame_start: int,
        frame_end: int
    ) -> List[np.ndarray]:
        """
        Reads sampled frames, applies safe crop, converts to RGB.

        Returns:
            List of RGB cropped frames. Length should be target_frames.
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(f"OpenCV could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        frame_indices = self.get_sample_indices(
            frame_start=frame_start,
            frame_end=frame_end,
            total_frames=total_frames
        )

        processed_frames = []

        for frame_idx in frame_indices:
            # Convert NSLT one-based frame number to OpenCV zero-based index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)

            ret, frame = cap.read()

            if not ret or frame is None:
                # Do not fail the whole video because of one bad frame.
                # Placeholder frame is RGB black.
                processed_frames.append(
                    np.zeros((224, 224, 3), dtype=np.uint8)
                )
                continue

            crop, crop_status = safe_crop(frame, bbox)

            if crop is None or crop.size == 0:
                crop = frame

            # MediaPipe expects RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            processed_frames.append(crop_rgb)

        cap.release()

        if len(processed_frames) != self.target_frames:
            raise ValueError(
                f"Invalid number of processed frames: {len(processed_frames)}"
            )

        return processed_frames


class MediaPipeExtractor:
    """
    Feature Extractor using MediaPipe Holistic.

    Output per frame:
    - Pose: 33 landmarks * 3 = 99
    - Left hand: 21 landmarks * 3 = 63
    - Right hand: 21 landmarks * 3 = 63

    Total feature vector size: 225.
    """

    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts a 225-dimensional feature vector from one RGB frame.
        If a body part is not detected, its landmarks are filled with zeros.
        If MediaPipe fails, returns a zero vector.
        """

        if frame is None or frame.size == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        frame.flags.writeable = False

        try:
            results = self.holistic.process(frame)
        except Exception:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        pose = np.zeros(33 * 3, dtype=np.float32)
        left_hand = np.zeros(21 * 3, dtype=np.float32)
        right_hand = np.zeros(21 * 3, dtype=np.float32)

        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                dtype=np.float32
            ).flatten()

        if results.left_hand_landmarks:
            left_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()

        if results.right_hand_landmarks:
            right_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()

        frame_features = np.concatenate([pose, left_hand, right_hand])

        if frame_features.shape != (FEATURE_DIM,):
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        return frame_features.astype(np.float32)

    def extract_features(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extracts MediaPipe features for all frames.

        Args:
            video_frames: List of RGB frames.

        Returns:
            np.ndarray with shape (30, 225).
        """

        features = []

        for frame in video_frames:
            frame_features = self.extract_frame_features(frame)
            features.append(frame_features)

        features = np.array(features, dtype=np.float32)

        if features.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(f"Invalid features shape: {features.shape}")

        return features

    def close(self):
        self.holistic.close()


def process_dataset(
    wlasl_json_path: str,
    nslt_json_path: str,
    videos_dir: str,
    output_dir: str,
    limit: int = None
):
    """
    Main preprocessing pipeline.
    """

    print("Loading metadata...", flush=True)

    print(f"WLASL JSON path: {wlasl_json_path}", flush=True)
    print(f"NSLT JSON path: {nslt_json_path}", flush=True)
    print(f"Videos dir: {videos_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    print(f"WLASL exists: {os.path.exists(wlasl_json_path)}", flush=True)
    print(f"NSLT exists: {os.path.exists(nslt_json_path)}", flush=True)
    print(f"Videos dir exists: {os.path.exists(videos_dir)}", flush=True)

    features_dir = os.path.join(output_dir, "features_mp")
    os.makedirs(features_dir, exist_ok=True)

    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(features_dir, subset), exist_ok=True)

    metadata_csv_path = os.path.join(output_dir, "metadata_mp.csv")
    error_log_path = os.path.join(output_dir, "preprocess_errors.log")

    wlasl_data = load_wlasl_metadata(wlasl_json_path)
    nslt_data = load_nslt_subset(nslt_json_path)

    print(f"WLASL entries loaded: {len(wlasl_data)}", flush=True)
    print(f"NSLT videos loaded: {len(nslt_data)}", flush=True)

    print("Initializing VideoPreprocessor...", flush=True)
    preprocessor = VideoPreprocessor(target_frames=SEQUENCE_LENGTH)

    print("Initializing MediaPipeExtractor...", flush=True)
    extractor = MediaPipeExtractor()
    print("MediaPipeExtractor initialized successfully.", flush=True)

    processed_count = 0
    error_count = 0
    missing_count = 0
    skipped_existing_count = 0

    error_log = []
    metadata_rows = []

    print("Starting video processing loop...", flush=True)

    for idx, video_id in enumerate(tqdm(nslt_data.keys())):
        video_id = normalize_video_id(video_id)

        if limit is not None and processed_count >= limit:
            print(f"\nLimit reached: {limit} processed videos.", flush=True)
            break

        if idx < 10 or idx % 100 == 0:
            print(f"\nProcessing index={idx}, video_id={video_id}", flush=True)

        try:
            video_info = get_video_info(
                video_id=video_id,
                wlasl_data=wlasl_data,
                nslt_data=nslt_data
            )

            subset = video_info["subset"]
            action_label = int(video_info["action_label"])

            video_path = find_video_path(
                videos_dir=videos_dir,
                video_id=video_id,
                subset=subset
            )

            if video_path is None:
                missing_count += 1

                if idx < 10 or idx % 100 == 0:
                    print(
                        f"[MISSING] video_id={video_id}, subset={subset}",
                        flush=True
                    )

                continue

            feature_path = os.path.join(
                features_dir,
                subset,
                f"{video_id}.npy"
            )

            # Avoid reprocessing already-created tensors
            if os.path.exists(feature_path):
                skipped_existing_count += 1

                metadata_rows.append({
                    "video_id": video_id,
                    "subset": subset,
                    "action_label": action_label,
                    "gloss": video_info["gloss"],
                    "feature_path": feature_path
                })

                if idx < 10 or idx % 100 == 0:
                    print(
                        f"[SKIP EXISTING] video_id={video_id}, path={feature_path}",
                        flush=True
                    )

                continue

            processed_frames = preprocessor.preprocess_video(
                video_path=video_path,
                bbox=video_info["bbox"],
                frame_start=int(video_info["frame_start"]),
                frame_end=int(video_info["frame_end"])
            )

            features = extractor.extract_features(processed_frames)

            if features.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                raise ValueError(f"Invalid features shape: {features.shape}")

            np.save(feature_path, features)

            metadata_rows.append({
                "video_id": video_id,
                "subset": subset,
                "action_label": action_label,
                "gloss": video_info["gloss"],
                "feature_path": feature_path
            })

            processed_count += 1

            if idx < 10 or idx % 100 == 0:
                print(
                    f"[OK] video_id={video_id}, features_shape={features.shape}",
                    flush=True
                )

        except Exception as e:
            error_msg = f"Error processing {video_id}: {repr(e)}"
            error_log.append(error_msg)
            error_count += 1

            if idx < 10 or idx % 100 == 0:
                print(f"[ERROR] {error_msg}", flush=True)

    extractor.close()

    print("\nWriting metadata CSV...", flush=True)

    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "subset",
            "action_label",
            "gloss",
            "feature_path"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("Writing error log...", flush=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        for error in error_log:
            f.write(error + "\n")

    print("\nProcessing Complete.", flush=True)
    print(f"Success newly processed: {processed_count}", flush=True)
    print(f"Skipped existing tensors: {skipped_existing_count}", flush=True)
    print(f"Missing videos skipped: {missing_count}", flush=True)
    print(f"Errors: {error_count}", flush=True)
    print(f"Features directory: {features_dir}", flush=True)
    print(f"Metadata CSV: {metadata_csv_path}", flush=True)
    print(f"Error log: {error_log_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ASL videos with MediaPipe Holistic"
    )

    parser.add_argument("--wlasl_json", type=str, required=True)
    parser.add_argument("--nslt_json", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging. Example: --limit 10"
    )

    args = parser.parse_args()

    process_dataset(
        wlasl_json_path=args.wlasl_json,
        nslt_json_path=args.nslt_json,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        limit=args.limit
    )


if __name__ == "__main__":
    main()