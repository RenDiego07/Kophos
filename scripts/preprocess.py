"""
ASL Video Preprocessing Script (MediaPipe Edition)
- Metadata loading
- Largest Bounding Box cropping (No Resizing)
- Temporal sampling (30 frames)
- Spatial Feature Extraction with MediaPipe Holistic
"""

import os
import cv2
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import mediapipe as mp

import torch
import torch.nn as nn

# Ensure you have these functions defined in your utils.py
from utils import (
    load_wlasl_metadata,
    load_nslt_subset,
    get_video_info,
    print_dataset_stats
)

class VideoPreprocessor:
    """
    Class to preprocess ASL videos following this pipeline:
    1. Load video frames.
    2. Apply bounding box (crop) to isolate signer.
    3. Temporal sampling to exactly 30 frames.
    """
    
    def __init__(self, target_frames: int = 30):
        self.target_frames = target_frames
    
    def load_video(self, video_path: str) -> List[np.ndarray]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for MediaPipe
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def crop_frame(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        ymin, xmin, ymax, xmax = bbox
        h, w = frame.shape[:2]
        
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        
        return frame[ymin:ymax, xmin:xmax]
    
    def temporal_sampling(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        num_frames = len(frames)
        
        if num_frames == 0:
            return []
            
        if num_frames == self.target_frames:
            return frames
        elif num_frames > self.target_frames:
            indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
            return [frames[i] for i in indices]
        else:
            padded_frames = frames.copy()
            last_frame = frames[-1]
            for _ in range(self.target_frames - num_frames):
                padded_frames.append(last_frame.copy())
            return padded_frames
    
    def preprocess_video(self, video_path: str, bbox: List[int]) -> List[np.ndarray]:
        frames = self.load_video(video_path)
        
        if not frames:
            raise ValueError(f"Could not load frames from: {video_path}")
        
        # Crop frames without resizing to maintain aspect ratio for MediaPipe
        cropped_frames = [self.crop_frame(frame, bbox) for frame in frames]
        return self.temporal_sampling(cropped_frames)

class MediaPipeExtractor:
    """
    Feature Extractor using MediaPipe Holistic
    """
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_features(self, video_frames: List[np.ndarray]) -> np.ndarray:
        features = []
        
        for frame in video_frames:
            # Optimize performance by marking image as not writeable
            frame.flags.writeable = False
            results = self.holistic.process(frame)
            
            # Pose: 33 points * 3 (x,y,z) = 99
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
            # Left Hand: 21 points * 3 = 63
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
            # Right Hand: 21 points * 3 = 63
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
            
            frame_features = np.concatenate([pose, lh, rh])
            features.append(frame_features)
            
        return np.array(features) # Shape: (30, 225)

def process_dataset(wlasl_json_path: str, nslt_json_path: str, videos_dir: str, output_dir: str):
    features_dir = os.path.join(output_dir, 'features_mp')
    os.makedirs(features_dir, exist_ok=True)
    
    print("Loading metadata...")
    wlasl_data = load_wlasl_metadata(wlasl_json_path)
    nslt_data = load_nslt_subset(nslt_json_path)
    
    preprocessor = VideoPreprocessor(target_frames=30)
    extractor = MediaPipeExtractor()
    
    processed_count = 0
    error_count = 0
    error_log = []
    
    for video_id in tqdm(nslt_data.keys()):
        try:
            video_info = get_video_info(video_id, wlasl_data, nslt_data)
            subset_folder  = video_info['subset']
            video_path = os.path.join(videos_dir, subset_folder, f"{video_id}.mp4")            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_id}")
            
            # 1. Preprocess (Crop & Temporal Sample)
            processed_frames = preprocessor.preprocess_video(video_path, video_info['bbox'])
            
            # 2. Extract Features with MediaPipe
            features = extractor.extract_features(processed_frames)
            
            # 3. Save feature tensor and metadata
            output_data = {
                'features': features,
                'video_id': video_id,
                'action_label': video_info['action_label'],
                'subset': video_info['subset'],
                'gloss': video_info['gloss']
            }
            
            np.save(os.path.join(features_dir, f"{video_id}.npy"), output_data)
            processed_count += 1
            
        except Exception as e:
            error_log.append(f"Error processing {video_id}: {str(e)}")
            error_count += 1
            
    print(f"\nProcessing Complete. Success: {processed_count}, Errors: {error_count}")

class BiLSTMSignModel(nn.Module):
    def __init__(self, input_size=225, hidden_size=256, num_layers=2, num_classes=100):
        super(BiLSTMSignModel, self).__init__()
        
        # input_size modified to 225 to match MediaPipe extraction (Pose + Left Hand + Right Hand)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocesar videos de ASL con MediaPipe')
    parser.add_argument('--wlasl_json', type=str, required=True)
    parser.add_argument('--nslt_json', type=str, required=True)
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    process_dataset(args.wlasl_json, args.nslt_json, args.videos_dir, args.output_dir)