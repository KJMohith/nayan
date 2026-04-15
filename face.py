"""
main.py
-------
Real-time facial recognition from webcam using DeepFace embeddings.

Workflow:
    1. Add photos:    data/<FriendName>/photo.jpg
    2. Encode:        python encode_faces.py
    3. Recognise:     python main.py

Controls:
    Q  – quit
    E  – reload encodings on the fly (after adding new faces)
    S  – toggle info panel
"""

import os
import sys
import pickle
import time

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import verification   # for cosine distance helper

# ── Configuration ────────────────────────────────────────────────────────────
ENCODINGS_FILE  = "encodings.pkl"
CAMERA_INDEX    = 0          # 0 = default webcam; change to 1/2 for external
MODEL_NAME      = "VGG-Face" # Must match encode_faces.py
DETECTOR        = "opencv"   # opencv is fastest; try retinaface for accuracy
THRESHOLD       = 0.40       # cosine distance threshold (lower = stricter)
PROCESS_EVERY_N = 3          # analyse every Nth frame for speed
FONT            = cv2.FONT_HERSHEY_DUPLEX
UNKNOWN_LABEL   = "Unknown"

# Colours (BGR)
COL = {
    "green":  (0,   210, 90),
    "red":    (30,  30,  220),
    "white":  (240, 240, 240),
    "dark":   (15,  15,  15),
    "yellow": (0,   210, 230),
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


def load_encodings(path: str):
    if not os.path.isfile(path):
        print(f"[ERROR] '{path}' not found. Run:  python encode_faces.py")
        sys.exit(1)

    with open(path, "rb") as f:
        data = pickle.load(f)

    embeddings = [np.array(e) for e in data["embeddings"]]
    names      = data["names"]
    model      = data.get("model", MODEL_NAME)

    print(f"[INFO] Loaded {len(embeddings)} embedding(s) for "
          f"{sorted(set(names))}  (model: {model})")
    return embeddings, names


def find_match(query_embedding: np.ndarray,
               known_embeddings: list,
               known_names: list) -> tuple[str, float]:
    """Return (best_name, distance). Returns UNKNOWN_LABEL if above threshold."""
    if not known_embeddings:
        return UNKNOWN_LABEL, 1.0

    distances = [cosine_distance(query_embedding, e) for e in known_embeddings]
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]

    if best_dist <= THRESHOLD:
        return known_names[best_idx], best_dist
    return UNKNOWN_LABEL, best_dist


def get_embeddings_from_frame(frame: np.ndarray) -> list[dict]:
    """
    Run DeepFace on the frame.
    Returns list of dicts: {"region": {x,y,w,h}, "embedding": np.ndarray}
    """
    try:
        results = DeepFace.represent(
            img_path          = frame,
            model_name        = MODEL_NAME,
            detector_backend  = DETECTOR,
            enforce_detection = True,
        )
        output = []
        for r in results:
            output.append({
                "region":    r["facial_area"],     # {x, y, w, h}
                "embedding": np.array(r["embedding"]),
            })
        return output
    except Exception:
        return []   # no face detected → silent


def draw_face_box(frame, region: dict, name: str, distance: float):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    is_known   = name != UNKNOWN_LABEL
    box_col    = COL["green"] if is_known else COL["red"]

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_col, 2)

    # Label strip below box
    label      = f"{name}  ({distance:.2f})" if is_known else UNKNOWN_LABEL
    strip_y1   = y + h
    strip_y2   = y + h + 30
    cv2.rectangle(frame, (x, strip_y1), (x + w, strip_y2), box_col, cv2.FILLED)
    cv2.putText(frame, label, (x + 5, strip_y2 - 8),
                FONT, 0.55, COL["dark"], 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, face_count: int, show_info: bool):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 65), COL["dark"], cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS   : {fps:5.1f}",       (10, 22), FONT, 0.55, COL["white"], 1)
    cv2.putText(frame, f"Faces : {face_count}",      (10, 44), FONT, 0.55, COL["white"], 1)
    cv2.putText(frame, f"Model : {MODEL_NAME}",      (10, 66), FONT, 0.45, COL["yellow"], 1)

    if show_info:
        h, w = frame.shape[:2]
        tips = ["Q = Quit", "E = Reload encodings", "S = Hide tips"]
        for i, tip in enumerate(tips):
            cv2.putText(frame, tip, (w - 200, 24 + i * 22),
                        FONT, 0.45, COL["yellow"], 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────

def run():
    known_embeddings, known_names = load_encodings(ENCODINGS_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Camera live. Press Q to quit.")

    frame_count  = 0
    face_results = []   # cached: list of (region, name, distance)
    fps          = 0.0
    prev_time    = time.time()
    show_info    = True

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── Run DeepFace every Nth frame ──────────────────────────────────
        if frame_count % PROCESS_EVERY_N == 0:
            detections   = get_embeddings_from_frame(frame)
            face_results = []

            for det in detections:
                name, dist = find_match(det["embedding"],
                                        known_embeddings, known_names)
                face_results.append((det["region"], name, dist))

        # ── Draw ──────────────────────────────────────────────────────────
        for region, name, dist in face_results:
            draw_face_box(frame, region, name, dist)

        draw_hud(frame, fps, len(face_results), show_info)

        cv2.imshow("DeepFace Facial Recognition", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit.")
            break
        elif key == ord("e"):
            print("[INFO] Reloading encodings...")
            known_embeddings, known_names = load_encodings(ENCODINGS_FILE)
        elif key == ord("s"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run()