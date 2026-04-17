import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace

ENCODINGS_FILE = "encodings.pkl"

MODEL_NAME = "Facenet512"

THRESHOLD = 0.40

UNKNOWN = "Unknown"


with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_embs = [np.array(e) for e in data["embeddings"]]
known_names = data["names"]

print("Loaded:", set(known_names))


# ---------------- Face Detector ----------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- Cosine Distance ----------------

def cosine_dist(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return 1 - np.dot(a, b)


def find_match(query):

    best_name = UNKNOWN
    best_dist = 999

    for emb, name in zip(known_embs, known_names):

        dist = cosine_dist(query, emb)

        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist < THRESHOLD:
        return best_name
    else:
        return UNKNOWN


# ---------------- Camera ----------------

cap = cv2.VideoCapture(0)

frame_count = 0
current_name = UNKNOWN

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # run recognition every 10 frames
        if frame_count % 10 == 0:

            face = frame[y:y+h, x:x+w]

            try:

                rep = DeepFace.represent(
                    img_path=face,
                    model_name=MODEL_NAME,
                    detector_backend="skip"
                )

                emb = np.array(rep[0]["embedding"])

                current_name = find_match(emb)

            except:
                pass

        color = (0,200,0) if current_name != UNKNOWN else (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.rectangle(frame,(x,y+h),(x+w,y+h+25),color,-1)

        cv2.putText(
            frame,
            current_name,
            (x+5,y+h+18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )

    frame_count += 1

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()