def debug_video_paths(
    nslt_data: Dict,
    videos_dir: str,
    num_videos: int = 10
) -> None:
    """
    Verifica si los videos existen físicamente en train/val/test.

    Args:
        nslt_data: Diccionario cargado desde nslt_100.json.
        videos_dir: Carpeta raíz videos_nslt_100.
        num_videos: Número de videos a revisar.
    """
    print("=" * 80)
    print("DEBUG VIDEO PATHS")
    print("=" * 80)

    checked = 0
    found = 0
    missing = 0

    for video_id, info in list(nslt_data.items())[:num_videos]:
        video_id = normalize_video_id(video_id)
        subset = info["subset"]

        video_path = find_video_path(
            videos_dir=videos_dir,
            video_id=video_id,
            subset=subset
        )

        exists = video_path is not None

        print(f"\nvideo_id: {video_id}")
        print(f"subset: {subset}")
        print(f"path: {video_path}")
        print(f"exists: {exists}")

        checked += 1

        if exists:
            found += 1
        else:
            missing += 1

    print("\n" + "=" * 80)
    print("RESUMEN PATHS")
    print("=" * 80)
    print(f"Revisados: {checked}")
    print(f"Encontrados: {found}")
    print(f"Faltantes: {missing}")
    print("=" * 80)




def debug_video_extraction(
    wlasl_data: List[Dict],
    nslt_data: Dict,
    videos_dir: str,
    num_videos: int = 5,
    num_frames_per_video: int = 3
) -> None:
    """
    Debuggea la extracción básica antes de correr todo el pipeline.

    Revisa:
    - video_path
    - bbox
    - frame.shape
    - crop.shape
    - rango de frames
    - si OpenCV puede abrir el video

    Args:
        wlasl_data: Metadata de WLASL.
        nslt_data: Metadata de NSLT-100.
        videos_dir: Carpeta raíz videos_nslt_100.
        num_videos: Número de videos a probar.
        num_frames_per_video: Número de frames a revisar por video.
    """
    print("=" * 80)
    print("DEBUG VIDEO EXTRACTION")
    print("=" * 80)

    video_index = build_wlasl_video_index(wlasl_data)

    debug_ids = list(nslt_data.keys())[:num_videos]

    for video_id in debug_ids:
        video_id = normalize_video_id(video_id)

        print("\n" + "=" * 80)
        print(f"VIDEO ID: {video_id}")

        if video_id not in nslt_data:
            print("[ERROR] Video no encontrado en NSLT")
            continue

        nslt_info = nslt_data[video_id]

        subset = nslt_info["subset"]
        action = nslt_info["action"]

        action_label = action[0]
        frame_start = action[1]
        frame_end = action[2]

        print(f"subset: {subset}")
        print(f"action_label: {action_label}")
        print(f"frame_start: {frame_start}")
        print(f"frame_end: {frame_end}")

        if video_id not in video_index:
            print("[ERROR] Video no encontrado en WLASL")
            continue

        wlasl_instance = video_index[video_id]["instance"]
        gloss = video_index[video_id]["gloss"]
        bbox = wlasl_instance["bbox"]

        print(f"gloss: {gloss}")
        print(f"bbox: {bbox}  # [x_min, y_min, x_max, y_max]")

        video_path = find_video_path(
            videos_dir=videos_dir,
            video_id=video_id,
            subset=subset
        )

        print(f"video_path: {video_path}")
        print(f"exists: {video_path is not None}")

        if video_path is None:
            print("[ERROR] No se encontró el video en train/val/test")
            continue

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("[ERROR] OpenCV no pudo abrir el video")
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"total_frames: {total_frames}")
        print(f"fps: {fps}")
        print(f"width: {width}")
        print(f"height: {height}")

        if frame_start < 1:
            frame_start = 1

        if frame_end == -1 or frame_end > total_frames:
            frame_end = total_frames

        if frame_start >= frame_end:
            print("[ERROR] Rango de frames inválido después de ajustar")
            cap.release()
            continue

        test_frame_indices = np.linspace(
            frame_start,
            frame_end - 1,
            num_frames_per_video
        ).astype(int)

        print(f"test_frame_indices: {test_frame_indices}")

        for frame_idx in test_frame_indices:
            print("\n--- FRAME DEBUG ---")
            print(f"frame_idx: {frame_idx}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)

            ret, frame = cap.read()

            print(f"ret: {ret}")

            if not ret or frame is None:
                print("[ERROR] No se pudo leer el frame")
                continue

            print(f"frame.shape: {frame.shape}")

            crop, crop_status = safe_crop(frame, bbox)

            print(f"crop_status: {crop_status}")
            print(f"crop.shape: {crop.shape}")

            if crop.shape[0] <= 0 or crop.shape[1] <= 0:
                print("[ERROR] Crop inválido")
            else:
                print("[OK] Crop válido")

        cap.release()

    print("\n" + "=" * 80)
    print("DEBUG FINALIZADO")
    print("=" * 80)





def debug_mediapipe_extraction(
    wlasl_data,
    nslt_data,
    videos_dir,
    num_videos=5,
    num_frames_per_video=3
):
    print("=" * 80)
    print("DEBUG MEDIAPIPE EXTRACTION")
    print("=" * 80)

    video_index = build_wlasl_video_index(wlasl_data)
    mp_holistic = mp.solutions.holistic

    tested_videos = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for video_id, nslt_info in nslt_data.items():
            video_id = normalize_video_id(video_id)
            subset = nslt_info["subset"]

            video_path = find_video_path(
                videos_dir=videos_dir,
                video_id=video_id,
                subset=subset
            )

            # Saltar videos faltantes
            if video_path is None:
                continue

            if video_id not in video_index:
                continue

            print("\n" + "=" * 80)
            print("VIDEO ID:", video_id)
            print("subset:", subset)
            print("video_path:", video_path)

            bbox = video_index[video_id]["instance"]["bbox"]
            gloss = video_index[video_id]["gloss"]

            frame_start = nslt_info["action"][1]
            frame_end = nslt_info["action"][2]

            print("gloss:", gloss)
            print("bbox:", bbox)
            print("frame_start:", frame_start)
            print("frame_end:", frame_end)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("[ERROR] OpenCV no pudo abrir el video")
                cap.release()
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_start < 1:
                frame_start = 1

            if frame_end == -1 or frame_end > total_frames:
                frame_end = total_frames

            if frame_start >= frame_end:
                print("[ERROR] Rango de frames inválido")
                cap.release()
                continue

            test_frame_indices = np.linspace(
                frame_start,
                frame_end - 1,
                num_frames_per_video
            ).astype(int)

            for frame_idx in test_frame_indices:
                print("\n--- FRAME:", frame_idx, "---")

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("[ERROR] No se pudo leer frame")
                    continue

                crop, crop_status = safe_crop(frame, bbox)

                print("frame.shape:", frame.shape)
                print("crop_status:", crop_status)
                print("crop.shape:", crop.shape)

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                try:
                    results = holistic.process(crop_rgb)

                    features = extract_landmarks_from_results(results)

                    print("pose detected:", results.pose_landmarks is not None)
                    print("left hand detected:", results.left_hand_landmarks is not None)
                    print("right hand detected:", results.right_hand_landmarks is not None)
                    print("features.shape:", features.shape)

                    if features.shape == (225,):
                        print("[OK] Feature vector correcto")
                    else:
                        print("[ERROR] Feature vector incorrecto")

                except Exception as e:
                    print("[ERROR MEDIAPIPE]", repr(e))

            cap.release()

            tested_videos += 1

            if tested_videos >= num_videos:
                break

    print("\n" + "=" * 80)
    print("DEBUG MEDIAPIPE FINALIZADO")
    print("=" * 80)




def extract_landmarks_from_results(results):
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

    features = np.concatenate([pose, left_hand, right_hand])

    return features

