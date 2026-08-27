"""
Aplica flip horizontal a todos los videos en ASL_Citizen 2/videos/
y genera asl_citizen_index.json con originales + flips.
Split: 75% train, 12.5% val, 12.5% test por clase.
"""
import os
import json
import random
import cv2
from collections import defaultdict

CLASSES = [
    "BAD", "BECOME", "FIND", "FINISH", "GOOD", "HELLO", "HELP", "HOSPITAL",
    "KNOW", "LAUGH", "LIKE", "MAKE", "ME", "MORE", "NEED", "NO", "PLAY",
    "PLEASE", "READ", "SHOW", "START", "TELL", "THINK", "WANT", "WHERE",
    "WHO", "WHY", "YES", "YOU"
]
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(ROOT_DIR, "data", "ASL_Citizen 2", "videos")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "asl_citizen_29_index.json")

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.125
# test = 1 - TRAIN_RATIO - VAL_RATIO


def assign_subsets(videos: list[str], seed: int = 42) -> dict[str, str]:
    """Asigna subset a cada video de forma estratificada por clase."""
    by_class = defaultdict(list)
    for v in videos:
        cls = v.split("-", 1)[1].replace("_flip", "").replace(".mp4", "")
        by_class[cls].append(v)

    subset_map = {}
    rng = random.Random(seed)
    for cls, vids in by_class.items():
        shuffled = vids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * TRAIN_RATIO))
        n_val   = max(1, round(n * VAL_RATIO))
        for i, v in enumerate(shuffled):
            if i < n_train:
                subset_map[v] = "train"
            elif i < n_train + n_val:
                subset_map[v] = "val"
            else:
                subset_map[v] = "test"
    return subset_map


def flip_video(src: str, dst: str) -> bool:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {src}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst, fourcc, fps, (width, height))

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
        written += 1

    cap.release()
    out.release()
    return written > 0


def main():
    originals = [f for f in os.listdir(VIDEOS_DIR)
                 if f.endswith(".mp4") and "_flip" not in f]
    originals.sort()
    print(f"Videos originales encontrados: {len(originals)}")

    # --- Generar flips ---
    flips_created = 0
    flip_names = []
    for filename in originals:
        stem = filename[:-4]                          # sin .mp4
        flip_name = f"{stem}_flip.mp4"
        src = os.path.join(VIDEOS_DIR, filename)
        dst = os.path.join(VIDEOS_DIR, flip_name)

        if os.path.exists(dst):
            print(f"  [SKIP] Ya existe: {flip_name}")
        else:
            print(f"  Generando flip: {flip_name}")
            if flip_video(src, dst):
                flips_created += 1
            else:
                print(f"  [ERROR] Falló: {filename}")
                continue

        flip_names.append(flip_name)

    print(f"\nFlips generados: {flips_created}  (ya existían: {len(flip_names) - flips_created})")

    # --- Asignar subsets (solo a originales; flips heredan el mismo subset) ---
    subset_map_orig = assign_subsets(originals)

    # --- Construir índice ---
    index = {}

    for filename in originals:
        key    = filename[:-4]
        cls    = key.split("-", 1)[1]
        subset = subset_map_orig[filename]
        if cls not in CLASS_INDEX:
            print(f"  [WARN] Clase desconocida: {cls} ({filename})")
            continue
        index[key] = {"subset": subset, "action": [CLASS_INDEX[cls], 1, -1]}

    for flip_name in flip_names:
        flip_key  = flip_name[:-4]
        orig_key  = flip_key.replace("_flip", "")
        cls_flip  = flip_key.split("-", 1)[1].replace("_flip", "")
        if cls_flip not in CLASS_INDEX:
            continue
        orig_entry = index.get(orig_key)
        subset = orig_entry["subset"] if orig_entry else "train"
        index[flip_key] = {"subset": subset, "action": [CLASS_INDEX[cls_flip], 1, -1]}

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    from collections import Counter
    subsets = Counter(v["subset"] for v in index.values())
    print(f"\nÍndice guardado en: {INDEX_PATH}")
    print(f"Total entradas : {len(index)}")
    print(f"  train : {subsets['train']}")
    print(f"  val   : {subsets['val']}")
    print(f"  test  : {subsets['test']}")


if __name__ == "__main__":
    main()
