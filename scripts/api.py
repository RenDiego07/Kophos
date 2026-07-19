import os
import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Parche para importar desde el directorio raíz si estás en /scripts

from model import BiLSTMSignModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Kophos MVP API", description="API de Inferencia para Lenguaje de Señas")

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],  # para pruebas rápidas

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],

)

# 1. Configuración de Parámetros Globales
MODEL_PATH = "/Users/dfflores/Downloads/best_bilstm_model.pth" # Ajusta tu ruta
CLASSES = ["HOSPITAL", "LAUGH", "MAKE", "ME", "NEED"]
SEQUENCE_LENGTH = 30
FEATURE_DIM = 225

# 2. Inicializar Modelo BiLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cargando modelo en: {device}")

model = BiLSTMSignModel(input_dim=FEATURE_DIM, hidden_dim=128, num_classes=5)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 3. Inicializar MediaPipe
mp_holistic = mp.solutions.holistic

def extract_features(results):
    pose = np.zeros(33 * 3, dtype=np.float32)
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

def adjust_sequence_length(frames, target_length=30):
    """Fuerza la secuencia a tener exactamente 30 frames."""
    if len(frames) == target_length:
        return np.array(frames)
    elif len(frames) == 0:
        return np.zeros((target_length, FEATURE_DIM))
    indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)
    return np.array(frames)[indices]

# 4. Endpoint de Predicción
@app.post("/predict/")
async def predict_sign(video: UploadFile = File(...)):
    """
    Recibe un archivo de video, extrae los landmarks y devuelve la predicción.
    """
    # Guardar el video temporalmente para que OpenCV pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    recorded_frames = []
    
    # Abrir contexto de MediaPipe por cada petición
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar el frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            features = extract_features(results)
            recorded_frames.append(features)

    cap.release()
    os.remove(temp_video_path) # Limpiar archivo temporal

    # Lógica de Inferencia
    if len(recorded_frames) < 5:
        return JSONResponse(status_code=400, content={"error": "El video es demasiado corto."})

    processed_data = adjust_sequence_length(recorded_frames, target_length=SEQUENCE_LENGTH)
    input_tensor = torch.tensor(processed_data).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted_idx.item()]
    conf_value = float(confidence.item())

    # Respuesta JSON estructurada
    return {
        "prediction": predicted_class if conf_value > 0.60 else "Desconocido",
        "confidence": round(conf_value * 100, 2),
        "total_frames_processed": len(recorded_frames),
        "raw_probabilities": {CLASSES[i]: round(float(probabilities[0][i]) * 100, 2) for i in range(len(CLASSES))}
    }