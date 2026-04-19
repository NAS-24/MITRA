import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'database'))

import cv2
import numpy as np
from datetime import datetime

from matcher import load_db, save_db, get_embedding, find_match, extract_face
import db as _db

# Active patient — set via env or change directly
ACTIVE_PATIENT_ID: int = int(os.getenv("MITRA_PATIENT_ID", "1"))

# Load face DB once at startup
db = load_db(patient_id=ACTIVE_PATIENT_ID)


def process_image(frame, name, relationship, notes):
    """
    Main recognition pipeline.
    - Decodes frame bytes → detects face → generates embedding
    - Tries to match against MySQL-backed DB
    - KNOWN  → updates last_met, logs timeline, returns person card data
    - UNKNOWN with name provided → registers to DB, returns person card data
    - UNKNOWN with no name → signals frontend to show register form
    """
    name_unknown = name

    try:
        nparr = np.frombuffer(frame, np.uint8)
        if nparr.size == 0:
            return {"error": "Empty image"}

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Error: Unable to decode image", flush=True)
            return {"error": "Invalid image format"}

        face      = extract_face(frame)
        embedding = get_embedding(face)

        matched_name, score = find_match(embedding, db)

        # ── CASE 1: KNOWN PERSON ──────────────────────────────────────────────
        if matched_name and score > 0.875:
            print(f"[MATCH] {matched_name} ({score:.4f})", flush=True)

            person = db[matched_name]
            person["last_met"] = datetime.now().isoformat()
            save_db(db, patient_id=ACTIVE_PATIENT_ID)   # persists last_met to MySQL

            # Log to interaction_timeline
            _db.log_interaction(
                patient_id         = ACTIVE_PATIENT_ID,
                recognition_status = "recognized",
                person_uuid        = person.get("id"),
                confidence_score   = float(score),
            )

            return {
                "person_id":          person.get("id"),
                "name":               matched_name,
                "relationship":       person.get("relationship", "Unknown"),
                "last_met_timestamp": person.get("last_met"),
                "notes":              person.get("notes", ""),
                "confidence_score":   float(score),
                "is_unknown":         False,
                "preferred_language": "English"
            }

        # ── CASE 2: UNKNOWN PERSON ────────────────────────────────────────────
        else:
            print("[NEW FACE]", flush=True)
            print("DEBUG INPUT IN PIPELINE:", name_unknown, relationship, notes, flush=True)

            # Log unknown event to timeline
            _db.log_interaction(
                patient_id         = ACTIVE_PATIENT_ID,
                recognition_status = "unknown",
                confidence_score   = float(score),
            )

            # Queue embedding for caregiver review
            _db.queue_unknown_face(
                patient_id = ACTIVE_PATIENT_ID,
                embedding  = embedding,
            )

            # No name supplied → tell frontend to show the register form
            if not name_unknown:
                return {"error": "Unknown face detected. Send name to register."}

            # Name supplied → register directly into MySQL
            person_uuid = _db.register_person(
                patient_id   = ACTIVE_PATIENT_ID,
                name         = name_unknown,
                relationship = relationship or "Unknown",
                notes        = notes or "",
                embedding    = embedding,
            )
            now_iso = datetime.now().isoformat()

            # Keep local in-memory db in sync
            db[name_unknown] = {
                "id":           person_uuid,
                "embedding":    embedding,
                "relationship": relationship or "Unknown",
                "notes":        notes or "",
                "last_met":     now_iso,
            }

            print(f"[REGISTERED] {name_unknown} / UUID: {person_uuid}", flush=True)

            return {
                "person_id":          person_uuid,
                "name":               name_unknown,
                "relationship":       relationship or "Unknown",
                "last_met_timestamp": now_iso,
                "notes":              notes or "",
                "confidence_score":   0.0,
                "is_unknown":         True,
                "preferred_language": "English"
            }

    except Exception as e:
        print("Error:", e, flush=True)
        return {"error": str(e)}