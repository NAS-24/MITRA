import os
import json
import numpy as np
from deepface import DeepFace

DB_PATH = "face_db.json"

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

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

def find_match(embedding, db, threshold=0.6):
    best_match = None
    best_score = -1

    for name, stored in db.items():
        score = cosine_similarity(embedding, stored)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > threshold:
        return best_match, best_score

    return None, None