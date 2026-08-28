import os
import sys
import cv2
import glob
import torch
import numpy as np
import mediapipe as mp
import tempfile
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from model import BiLSTMSignModel

load_dotenv()  # Cargar variables de entorno desde .env
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
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_bilstm_model-6.pth")
DATA_CLASSES_DIR  = os.path.join(ROOT_DIR, "data", "ASL_34_Classes")
DATA_FLAT_DIR     = os.path.join(ROOT_DIR, "data", "ASL_34_Flat_Videos")
CLASSES = [
    "BAD", "BECOME", "FIND", "FINISH", "GOOD", "HELLO", "HELP", "HOSPITAL",
    "KNOW", "LAUGH", "LIKE", "MAKE", "ME", "MORE", "NEED", "NO",
    "PLAY", "PLEASE", "READ", "SHOW", "START", "TELL", "THINK", "WANT",
    "WHERE", "WHO", "WHY", "YES", "YOU"
]
SEQUENCE_LENGTH = 30
FEATURE_DIM = 255  # 33*3 pose + 21*3 left_hand + 21*3 right_hand + 15 left_angles + 15 right_angles

_ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

def compute_hand_angles(landmarks_flat: np.ndarray) -> np.ndarray:
    if np.all(landmarks_flat == 0):
        return np.zeros(15, dtype=np.float32)
    pts = landmarks_flat.reshape(21, 3)
    angles = []
    for a, b, c in _ANGLE_TRIPLETS:
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return np.array(angles, dtype=np.float32)

# 2. Inicializar Modelo BiLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cargando modelo en: {device}")

model = BiLSTMSignModel(input_dim=FEATURE_DIM, hidden_dim=128, num_classes=len(CLASSES))
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 3. Configuración LLM (Llama via HuggingFace Router)
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"

SYSTEM_PROMPT = """You are Kophos, a helpful, empathetic, and natural conversational AI assistant.
The user is communicating with you through a real-time sign language translation system. The text you receive is the translated output of their signs.

Your goal is to maintain a natural, flowing conversation. You MUST adhere to the following strict rules:

1. NO META-INSTRUCTIONS: NEVER break character to discuss the sign language interface itself. NEVER instruct the user on how to perform physical signs (e.g., absolutely do not say "Can you do the NEED sign", "Point your finger", or "Make a gesture"). Treat the user's input as if they simply typed or spoke it to you.
2. NATURAL RESPONSES: Respond directly to the meaning and intent of the user's message like a real human assistant would. If they say "I need to go to the hospital", respond with urgency and care (e.g., "I understand. Is it an emergency? Do you need me to call an ambulance?").
3. LIMITED VOCABULARY AWARENESS: The user is communicating using a very constrained and limited sign language vocabulary of 29 basic words/concepts: BAD, BECOME, FIND, FINISH, GOOD, HELLO, HELP, HOSPITAL, KNOW, LAUGH, LIKE, MAKE, ME, MORE, NEED, NO, PLAY, PLEASE, READ, SHOW, START, TELL, THINK, WANT, WHERE, WHO, WHY, YES, YOU.
4. GUIDE THE CONVERSATION: To help the user continue the conversation easily, ALWAYS end your response with a single, highly simplified follow-up question. Formulate your questions so they can be answered with basic concepts (Yes/No, basic needs, or simple actions). Avoid complex open-ended questions.

Act as a true conversational partner, focusing entirely on the context of the chat, not the medium of communication."""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ReformulateRequest(BaseModel):
    words: list[str]
    context: list[ChatMessage] = []

@app.post("/reformulate/")
async def reformulate(req: ReformulateRequest):
    if not req.words:
        return JSONResponse(status_code=400, content={"error": "No hay palabras."})

    raw = " ".join(req.words)

    context_block = ""
    if req.context:
        recent = req.context[-6:]  # últimos 6 turnos para no saturar el prompt
        lines = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)
        context_block = f"\n\nContexto de la conversación reciente:\n{lines}\n"

    system_content = (
        "Eres un reformulador de frases para una app de lenguaje de señas. "
        "El usuario te enviará palabras sueltas detectadas por el sistema (en inglés, en mayúsculas). "
        "Tu tarea es convertirlas en una oración natural y gramaticalmente correcta en español, "
        "tomando en cuenta el contexto de la conversación si se proporciona. "
        "Si hay contexto, interpreta las palabras como una respuesta o continuación coherente de esa conversación. "
        "Responde ÚNICAMENTE con la oración reformulada, sin explicaciones ni texto adicional."
        + context_block
    )

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": raw},
        ],
    )
    suggestion = completion.choices[0].message.content.strip()
    return {"suggestion": suggestion}

@app.post("/chat/")
async def chat(req: ChatRequest):
    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "No hay mensajes."})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )

    response_text = completion.choices[0].message.content
    return {"response": response_text}

# 4. Inicializar MediaPipe
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

    left_angles  = compute_hand_angles(left_hand)
    right_angles = compute_hand_angles(right_hand)
    return np.concatenate([pose, left_hand, right_hand, left_angles, right_angles]).astype(np.float32)

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
    suffix = ".webm" if video.content_type == "video/webm" else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
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


def _get_sample_videos(word: str, n: int = 2) -> list[str]:
    """Return up to n non-flip MP4 paths for the given word class."""
    # Prefer per-class folder (train split), fall back to flat folder
    class_train = os.path.join(DATA_CLASSES_DIR, word, "train")
    if os.path.isdir(class_train):
        candidates = [
            p for p in glob.glob(os.path.join(class_train, "*.mp4"))
            if "_flip" not in os.path.basename(p)
        ]
    else:
        candidates = [
            p for p in glob.glob(os.path.join(DATA_FLAT_DIR, f"*-{word}.mp4"))
            if "_flip" not in os.path.basename(p)
        ]
    candidates.sort()
    return candidates[:n]


@app.get("/lexicon/{word}/video/{index}")
async def lexicon_video(word: str, index: int):
    """Serve a sample video for a given sign word (index 0 or 1)."""
    word = word.upper()
    if word not in CLASSES:
        return JSONResponse(status_code=404, content={"error": "Palabra no reconocida."})
    videos = _get_sample_videos(word, n=2)
    if index >= len(videos):
        return JSONResponse(status_code=404, content={"error": f"No hay video con índice {index}."})
    return FileResponse(videos[index], media_type="video/mp4")