import os
import pickle
import numpy as np
from deepface import DeepFace

# ---------------- CONFIG ----------------

DATA_DIR = "data"
ENCODINGS_FILE = "encodings.pkl"

MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ----------------------------------------


def encode_image(img_path):

    try:

        reps = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )

        embeddings = []

        for r in reps:

            embeddings.append(np.array(r["embedding"]))

        return embeddings

    except Exception as e:

        print("Skipped:", img_path, "|", e)

        return []


def build_encodings():

    if not os.path.exists(DATA_DIR):

        print("ERROR: data folder not found")

        return


    known_embeddings = []
    known_names = []


    persons = sorted(os.listdir(DATA_DIR))


    print("Scanning dataset...")


    for person in persons:

        person_dir = os.path.join(DATA_DIR, person)

        if not os.path.isdir(person_dir):
            continue


        print("\nEncoding:", person)


        count = 0


        for file in os.listdir(person_dir):

            if not file.lower().endswith(SUPPORTED_EXT):
                continue


            path = os.path.join(person_dir, file)

            embeddings = encode_image(path)


            for emb in embeddings:

                known_embeddings.append(emb)

                known_names.append(person)

                count += 1

                print("OK:", file)


        print("Faces encoded:", count)


    if len(known_embeddings) == 0:

        print("ERROR: No faces found in dataset")

        return


    data = {
        "embeddings": known_embeddings,
        "names": known_names,
        "model": MODEL_NAME
    }


    with open(ENCODINGS_FILE, "wb") as f:

        pickle.dump(data, f)


    print("\nSUCCESS")
    print("Saved", len(known_embeddings), "embeddings")
    print("People:", set(known_names))
    print("Output file:", ENCODINGS_FILE)


if __name__ == "__main__":

    print("Building face encodings using", MODEL_NAME)

    build_encodings()