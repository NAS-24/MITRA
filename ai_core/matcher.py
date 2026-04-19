import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'database'))

import numpy as np
import cv2
from mtcnn import MTCNN
from deepface import DeepFace

# DB module replaces the old JSON helpers
import db as _db

# ── Patient context ───────────────────────────────────────────────────────────
ACTIVE_PATIENT_ID: int = int(os.getenv("MITRA_PATIENT_ID", "1"))

# ── Face detector (loaded once at import) ─────────────────────────────────────
detector = MTCNN()


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def extract_face(frame: np.ndarray) -> np.ndarray:
    """Detects first face in frame (BGR), crops and resizes to 160×160."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    if not results:
        raise Exception("No face detected")

    x, y, w, h = results[0]['box']
    x, y        = max(0, x), max(0, y)
    h_img, w_img, _ = frame.shape
    x2, y2      = min(x + w, w_img), min(y + h, h_img)
    face        = frame[y:y2, x:x2]

    if face.size == 0:
        raise Exception("Invalid face crop")

    return cv2.resize(face, (160, 160))


def get_embedding(face: np.ndarray) -> np.ndarray:
    """Returns a normalized 512-D ArcFace embedding."""
    result    = DeepFace.represent(
        img_path=face,
        model_name="ArcFace",
        enforce_detection=False
    )
    embedding = np.array(result[0]["embedding"])
    return normalize(embedding)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_match(embedding, db, threshold=0.9):
    best_match = None
    best_score = -1.0

    for name, data in db.items():
        db_emb = np.array(data["embedding"])
        score  = float(np.dot(embedding, db_emb))
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > threshold:
        return best_match, best_score
    return None, best_score


def build_response(name, score, threshold=0.9):
    if name:
        return {"person_id": name, "confidence_score": float(score), "is_unknown": False}
    return {"person_id": None, "confidence_score": float(score) if score else 0.0,
            "is_unknown": True}


# ── DB shim — same call signatures, now backed by MySQL ──────────────────────

def load_db(patient_id: int = None) -> dict:
    """Loads face DB from MySQL (replaces face_db.json read)."""
    return _db.load_db(patient_id=patient_id or ACTIVE_PATIENT_ID)


def save_db(db: dict, patient_id: int = None) -> None:
    """Saves face DB to MySQL (replaces face_db.json write)."""
    _db.save_db(db, patient_id=patient_id or ACTIVE_PATIENT_ID)