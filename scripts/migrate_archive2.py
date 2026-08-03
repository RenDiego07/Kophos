"""
Migración de Archive 2 → ASL_15_Classes, augmentación con flip,
actualización del JSON con los 26 índices correctos,
y renombrado de carpetas y JSON a ASL_25_*.
"""
import os
import json
import random
import shutil
import cv2

# ── Configuración ──────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARCHIVE_DIR   = os.path.join(DATA_DIR, "Archive 2")
CLASSES_DIR   = os.path.join(DATA_DIR, "ASL_15_Classes")
FLAT_DIR      = os.path.join(DATA_DIR, "ASL_15_Flat_Videos")
INDEX_PATH    = os.path.join(DATA_DIR, "asl_15_citizen_index.json")

NEW_CLASSES_DIR = os.path.join(DATA_DIR, "ASL_25_Classes")
NEW_FLAT_DIR    = os.path.join(DATA_DIR, "ASL_25_Flat_Videos")
NEW_INDEX_PATH  = os.path.join(DATA_DIR, "asl_25_citizen_index.json")

EXISTING_CLASSES = [
    "HOSPITAL", "LAUGH", "MAKE", "ME", "NEED", "READ", "SHOW", "START",
    "STOP", "TELL", "THINK", "TO", "UNDERSTAND", "WAIT", "WANT", "WRITE"
]
NEW_CLASSES = ["ANSWER", "ASK", "CANCEL", "CHAT", "CLOSE", "FIND", "HELP", "KNOW", "LIKE", "OPEN"]
ALL_CLASSES = sorted(EXISTING_CLASSES + NEW_CLASSES)
CLASS_INDEX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

SUBSETS = ["train", "val", "test"]
SUBSET_SPLITS = [0.80, 0.10, 0.10]  # para clases nuevas sin split previo


def generate_unique_id(existing_ids: set) -> str:
    while True:
        new_id = str(random.randint(10**15, 10**16 - 1))
        if new_id not in existing_ids:
            return new_id


def assign_subset(idx: int, total: int) -> str:
    ratio = idx / total
    if ratio < SUBSET_SPLITS[0]:
        return "train"
    elif ratio < SUBSET_SPLITS[0] + SUBSET_SPLITS[1]:
        return "val"
    return "test"


def flip_video(src: str, dst: str) -> bool:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"    [ERROR] No se pudo abrir: {src}")
        return False
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
        frames += 1
    cap.release()
    out.release()
    return frames > 0


# ── 1. Cargar índice existente ─────────────────────────────────────────────────
print("Cargando índice existente...")
with open(INDEX_PATH) as f:
    index = json.load(f)

existing_ids = set(k.split("-")[0] for k in index.keys())
print(f"  Entradas existentes: {len(index)}")

# ── 2. Actualizar índices de clases existentes en el JSON ─────────────────────
print("\nActualizando índices de clases existentes (16 → 26 clases)...")
updated = 0
for key, entry in index.items():
    old_label = entry["action"][0]
    # Inferir clase desde la key (formato: {id}-{CLASE} o {id}-{CLASE}_flip)
    raw = key.split("-", 1)[1].replace("_flip", "")
    new_label = CLASS_INDEX.get(raw)
    if new_label is not None and new_label != old_label:
        entry["action"][0] = new_label
        updated += 1
print(f"  Índices actualizados: {updated}")

# ── 3. Mover videos de Archive 2 → ASL_15_Classes ─────────────────────────────
print("\nMoviendo videos de Archive 2 → ASL_15_Classes...")
moved_total = 0
new_entries = {}

for cls in NEW_CLASSES:
    src_cls_dir = os.path.join(ARCHIVE_DIR, cls)
    dst_cls_dir = os.path.join(CLASSES_DIR, cls)
    os.makedirs(dst_cls_dir, exist_ok=True)

    videos = sorted([f for f in os.listdir(src_cls_dir) if f.endswith(".mp4") and "_flip" not in f])
    print(f"\n  [{cls}] {len(videos)} videos")

    for i, filename in enumerate(videos):
        src_path = os.path.join(src_cls_dir, filename)
        dst_path = os.path.join(dst_cls_dir, filename)
        shutil.move(src_path, dst_path)

        # Registrar en índice con subset asignado por proporción
        key = filename[:-4]  # sin .mp4
        subset = assign_subset(i, len(videos))
        new_entries[key] = {
            "subset": subset,
            "action": [CLASS_INDEX[cls], 1, -1]
        }
        moved_total += 1

print(f"\n  Total movidos: {moved_total} videos")

# ── 4. Flip augmentation de las clases nuevas ──────────────────────────────────
print("\nGenerando flips para las clases nuevas...")
flip_entries = {}
flipped_total = 0
errors = 0

for cls in NEW_CLASSES:
    cls_dir = os.path.join(CLASSES_DIR, cls)
    videos  = [f for f in os.listdir(cls_dir) if f.endswith(".mp4") and "_flip" not in f]
    print(f"\n  [{cls}] {len(videos)} originales")

    for filename in videos:
        src_path = os.path.join(cls_dir, filename)
        orig_key = filename[:-4]
        subset   = new_entries.get(orig_key, {}).get("subset", "train")

        new_id = generate_unique_id(existing_ids | set(new_entries.keys()) | set(flip_entries.keys()))
        existing_ids.add(new_id)

        dst_filename = f"{new_id}-{cls}_flip.mp4"
        dst_path     = os.path.join(cls_dir, dst_filename)

        print(f"    {filename} → {dst_filename}")
        if flip_video(src_path, dst_path):
            flip_entries[f"{new_id}-{cls}_flip"] = {
                "subset": subset,
                "action": [CLASS_INDEX[cls], 1, -1]
            }
            flipped_total += 1
        else:
            errors += 1

print(f"\n  Flips generados: {flipped_total} | Errores: {errors}")

# ── 5. Copiar nuevos videos (originales + flips) a Flat_Videos ─────────────────
print("\nCopiando nuevos videos a ASL_15_Flat_Videos...")
copied = 0
for cls in NEW_CLASSES:
    cls_dir = os.path.join(CLASSES_DIR, cls)
    for filename in os.listdir(cls_dir):
        if filename.endswith(".mp4"):
            shutil.copy2(os.path.join(cls_dir, filename), os.path.join(FLAT_DIR, filename))
            copied += 1
print(f"  Copiados: {copied} archivos")

# ── 6. Consolidar índice completo ──────────────────────────────────────────────
index.update(new_entries)
index.update(flip_entries)
print(f"\nEntradas en índice: {len(index)}")

# ── 7. Guardar índice como asl_25_citizen_index.json ──────────────────────────
with open(NEW_INDEX_PATH, "w") as f:
    json.dump(index, f, indent=2)
print(f"Índice guardado: {NEW_INDEX_PATH}")

# ── 8. Renombrar carpetas ──────────────────────────────────────────────────────
print("\nRenombrando carpetas...")
os.rename(CLASSES_DIR, NEW_CLASSES_DIR)
print(f"  {CLASSES_DIR} → {NEW_CLASSES_DIR}")
os.rename(FLAT_DIR, NEW_FLAT_DIR)
print(f"  {FLAT_DIR} → {NEW_FLAT_DIR}")

# ── 9. Resumen final ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"Clases totales     : {len(ALL_CLASSES)}")
print(f"Videos originales  : {moved_total} nuevos + existentes")
print(f"Videos con flip    : {flipped_total}")
print(f"Total en índice    : {len(index)}")
print(f"Carpeta clases     : ASL_25_Classes")
print(f"Carpeta flat       : ASL_25_Flat_Videos")
print(f"Índice JSON        : asl_25_citizen_index.json")
print(f"\nOrden de clases (para train.py --num_classes 26):")
for i, c in enumerate(ALL_CLASSES):
    print(f"  {i:2d}: {c}")
