import os
import json
import numpy as np
from deepface import DeepFace
import cv2
from mtcnn import MTCNN

detector = MTCNN()

import cv2
from mtcnn import MTCNN

detector = MTCNN()

def extract_face(frame):
    # 🔥 Convert BGR → RGB (CRITICAL FIX)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(rgb)

    if len(results) == 0:
        raise Exception("No face detected")

    x, y, w, h = results[0]['box']

    x = max(0, x)
    y = max(0, y)

    h_img, w_img, _ = frame.shape
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    face = frame[y:y2, x:x2]

    if face.size == 0:
        raise Exception("Invalid face crop")

    face = cv2.resize(face, (160, 160))

    return face

DB_PATH = "face_db.json"

def load_db():
    import json
    import os
    import numpy as np

    if not os.path.exists("face_db.json"):
        return {}

    with open("face_db.json", "r") as f:
        content = f.read().strip()
        if not content:
            return {}

        data = json.loads(content)

    for name in data:
        data[name]["embedding"] = np.array(data[name]["embedding"])

    return data


def save_db(db):
    import json

    serializable_db = {}
    for name, data in db.items():
        serializable_db[name] = {
            "id": data["id"],
            "embedding": data["embedding"].tolist(),
            "relationship": data["relationship"],
            "notes": data["notes"],
            "last_met": data["last_met"]
        }

    with open("face_db.json", "w") as f:
        json.dump(serializable_db, f)

def get_embedding(frame):
    result = DeepFace.represent(
        img_path=frame,
        model_name="ArcFace",
        enforce_detection=True
    )
    return result[0]["embedding"]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def find_match(embedding, db, threshold=0.7):
    best_match = None
    best_score = -1

    for name, data in db.items():
        db_embedding = np.array(data["embedding"])

        score = np.dot(embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_match = name

    if best_score > threshold:
        return best_match, best_score

    return None, best_score

def build_response(name, score, threshold=0.85):
    if name:
        return {
            "person_id": name,
            "confidence_score": float(score),
            "is_unknown": False
        }
    else:
        return {
            "person_id": None,
            "confidence_score": float(score) if score else 0.0,
            "is_unknown": True
        }