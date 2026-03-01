"""
Script de preprocesamiento para videos de ASL
- Carga de metadatos
- Identificación del Largest Bounding Box
- Recorte y redimensionamiento (224x224)
- Muestreo temporal (30 frames)
- Extracción de características con MobileNetV2
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import json
from tqdm import tqdm

# PyTorch y torchvision para MobileNetV2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utils import (
    load_wlasl_metadata,
    load_nslt_subset,
    get_video_info,
    print_dataset_stats
)


class VideoPreprocessor:
    """
    Clase para preprocesar videos de ASL siguiendo el pipeline:
    1. Cargar video
    2. Aplicar bounding box
    3. Redimensionar a 224x224
    4. Muestreo temporal a 30 frames
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 target_frames: int = 30):
        """
        Args:
            target_size: Tamaño objetivo (ancho, alto)
            target_frames: Número objetivo de frames
        """
        self.target_size = target_size
        self.target_frames = target_frames
    
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """
        Carga un video y retorna lista de frames
        
        Args:
            video_path: Ruta al archivo de video
        
        Returns:
            Lista de frames como arrays numpy (H, W, C)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convertir de BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def crop_frame(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Recorta un frame usando el bounding box
        
        Args:
            frame: Frame como array numpy (H, W, C)
            bbox: Bounding box [ymin, xmin, ymax, xmax]
        
        Returns:
            Frame recortado
        """
        ymin, xmin, ymax, xmax = bbox
        
        # Asegurar que los índices estén dentro de los límites
        h, w = frame.shape[:2]
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        
        cropped = frame[ymin:ymax, xmin:xmax]
        return cropped
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensiona un frame al tamaño objetivo
        
        Args:
            frame: Frame como array numpy
        
        Returns:
            Frame redimensionado
        """
        resized = cv2.resize(frame, self.target_size, 
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def temporal_sampling(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Muestrea o rellena frames para alcanzar el número objetivo
        
        Args:
            frames: Lista de frames
        
        Returns:
            Lista de frames con longitud target_frames
        """
        num_frames = len(frames)
        
        if num_frames == self.target_frames:
            return frames
        
        elif num_frames > self.target_frames:
            # Muestreo uniforme
            indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            return sampled_frames
        
        else:
            # Padding: repetir el último frame
            padded_frames = frames.copy()
            last_frame = frames[-1]
            for _ in range(self.target_frames - num_frames):
                padded_frames.append(last_frame.copy())
            return padded_frames
    
    def preprocess_video(self, video_path: str, bbox: List[int]) -> np.ndarray:
        """
        Pipeline completo de preprocesamiento para un video
        
        Args:
            video_path: Ruta al video
            bbox: Bounding box [ymin, xmin, ymax, xmax]
        
        Returns:
            Array numpy de shape (T, H, W, C) con T=target_frames
        """
        # 1. Cargar video
        frames = self.load_video(video_path)
        
        if len(frames) == 0:
            raise ValueError(f"No se pudieron cargar frames del video: {video_path}")
        
        # 2. Recortar y redimensionar cada frame
        processed_frames = []
        for frame in frames:
            cropped = self.crop_frame(frame, bbox)
            resized = self.resize_frame(cropped)
            processed_frames.append(resized)
        
        # 3. Muestreo temporal
        sampled_frames = self.temporal_sampling(processed_frames)
        
        # Convertir a array numpy
        video_array = np.array(sampled_frames)  # Shape: (T, H, W, C)
        
        return video_array

