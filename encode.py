"""
encode_faces.py
---------------
Scans the `data/` folder, generates DeepFace embeddings for every
photo found, and saves them to `encodings.pkl`.

Folder structure expected:
    data/
        Alice/
            photo1.jpg
            photo2.jpg
        Bob/
            img1.jpg
        ...

Run once (or whenever you add / change photos):
    python encode_faces.py
"""

import os
import pickle
import numpy as np
from deepface import DeepFace

DATA_DIR       = "data"
ENCODINGS_FILE = "encodings.pkl"
MODEL_NAME     = "VGG-Face"          # Options: VGG-Face, Facenet, Facenet512, ArcFace
DETECTOR       = "opencv"            # Options: opencv, retinaface, mtcnn, ssd
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image(image_path: str):
    """Return embedding vector for the first face found in image, or None."""
    try:
        result = DeepFace.represent(
            img_path   = image_path,
            model_name = MODEL_NAME,
            detector_backend = DETECTOR,
            enforce_detection = True,
        )
        # result is a list; take the first face's embedding
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"    [WARN] Could not encode {os.path.basename(image_path)}: {e}")
        return None


def build_encodings():
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] '{DATA_DIR}/' folder not found.")
        print("        Create it and add sub-folders named after each friend.")
        return

    persons = sorted([
        p for p in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, p))
    ])

    if not persons:
        print(f"[ERROR] '{DATA_DIR}/' is empty. Add friend folders with photos.")
        return

    known_embeddings = []
    known_names      = []

    for person_name in persons:
        person_dir = os.path.join(DATA_DIR, person_name)
        print(f"\n[INFO] Processing: {person_name}")
        count = 0

        for filename in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue

            image_path = os.path.join(person_dir, filename)
            embedding  = encode_image(image_path)

            if embedding is not None:
                known_embeddings.append(embedding)
                known_names.append(person_name)
                count += 1
                print(f"    [OK] {filename}")

        print(f"    → {count} face(s) encoded for {person_name}")

    if not known_embeddings:
        print("\n[ERROR] No faces encoded. Check that photos contain clear, visible faces.")
        return

    data = {
        "embeddings": known_embeddings,
        "names":      known_names,
        "model":      MODEL_NAME,
    }
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n[SUCCESS] {len(known_embeddings)} embedding(s) saved for "
          f"{len(set(known_names))} person(s) → '{ENCODINGS_FILE}'")


if __name__ == "__main__":
    print("=== DeepFace — Encoding Faces ===")
    build_encodings()