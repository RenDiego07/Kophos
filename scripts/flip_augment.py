"""
Genera versiones volteadas horizontalmente de todos los videos en ASL_34_Classes
y actualiza asl_34_citizen_index.json con las nuevas entradas.
También copia los videos volteados a ASL_34_Flat_Videos.

Maneja dos estructuras de carpetas:
  - Con subsets: ASL_34_Classes/{CLASS}/train|test|val/{id}-{CLASS}.mp4
  - Sin subsets: ASL_34_Classes/{CLASS}/{id}-{CLASS}.mp4
"""
import os
import json
import random
import shutil
import cv2

CLASSES = [
    "ASK", "BAD", "BECOME", "CHAT", "FIND", "FINISH", "GOOD", "HELLO",
    "HELP", "HOSPITAL", "KNOW", "LAUGH", "LIKE", "MAKE", "ME", "MORE",
    "NEED", "NO", "PLAY", "PLEASE", "READ", "SHOW", "START", "TELL",
    "THINK", "TO", "UNDERSTAND", "WANT", "WHERE", "WHO", "WHY", "WRITE",
    "YES", "YOU"
]
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLASSES_DIR = os.path.join(DATA_DIR, "ASL_34_Classes")
FLAT_DIR = os.path.join(DATA_DIR, "ASL_34_Flat_Videos")
INDEX_PATH = os.path.join(DATA_DIR, "asl_34_citizen_index.json")
SUBSETS = ["train", "test", "val"]


def generate_unique_id(existing_ids: set) -> str:
    while True:
        new_id = str(random.randint(10**15, 10**16 - 1))
        if new_id not in existing_ids:
            return new_id


def flip_video(src_path: str, dst_path: str) -> bool:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {src_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    frames_written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
        frames_written += 1

    cap.release()
    out.release()
    return frames_written > 0


def collect_videos(cls: str, index: dict) -> list[tuple[str, str]]:
    """
    Retorna lista de (video_path_absoluto, subset) para la clase dada.
    Maneja tanto carpetas con subsets como sin ellos.
    """
    cls_dir = os.path.join(CLASSES_DIR, cls)
    results = []

    has_subsets = any(
        os.path.isdir(os.path.join(cls_dir, s)) for s in SUBSETS
    )

    if has_subsets:
        for subset in SUBSETS:
            subset_dir = os.path.join(cls_dir, subset)
            if not os.path.isdir(subset_dir):
                continue
            for filename in os.listdir(subset_dir):
                if filename.endswith(".mp4") and "_flip" not in filename:
                    results.append((os.path.join(subset_dir, filename), subset))
    else:
        # Buscar subset en el JSON por nombre de archivo (sin extensión)
        for filename in os.listdir(cls_dir):
            if not filename.endswith(".mp4") or "_flip" in filename:
                continue
            key = filename[:-4]  # quitar .mp4
            entry = index.get(key)
            subset = entry["subset"] if entry else "train"
            results.append((os.path.join(cls_dir, filename), subset))

    return results


def main():
    os.makedirs(FLAT_DIR, exist_ok=True)

    with open(INDEX_PATH) as f:
        index = json.load(f)

    existing_ids = set(k.split("-")[0] for k in index.keys())
    new_entries = {}
    total = 0
    errors = 0

    for cls in CLASSES:
        cls_dir = os.path.join(CLASSES_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"[SKIP] Carpeta no encontrada: {cls_dir}")
            continue

        print(f"\n[{cls}]")
        videos = collect_videos(cls, index)

        for src_path, subset in videos:
            new_id = generate_unique_id(existing_ids | set(new_entries.keys()))
            existing_ids.add(new_id)

            dst_filename = f"{new_id}-{cls}_flip.mp4"
            dst_path = os.path.join(os.path.dirname(src_path), dst_filename)

            orig_name = os.path.basename(src_path)
            print(f"  {orig_name} ({subset}) -> {dst_filename}")

            success = flip_video(src_path, dst_path)

            if success:
                index_key = f"{new_id}-{cls}_flip"
                new_entries[index_key] = {
                    "subset": subset,
                    "action": [CLASS_INDEX[cls], 1, -1]
                }
                # Copiar al directorio plano
                shutil.copy2(dst_path, os.path.join(FLAT_DIR, dst_filename))
                total += 1
            else:
                errors += 1

    index.update(new_entries)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Videos generados : {total}")
    print(f"Errores          : {errors}")
    print(f"Nuevas entradas  : {len(new_entries)}")
    print(f"Total en índice  : {len(index)}")


if __name__ == "__main__":
    main()
