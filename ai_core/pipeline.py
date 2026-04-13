import cv2
from matcher import load_db, save_db, get_embedding, find_match
from datetime import datetime
import uuid
import requests
from matcher import extract_face
import numpy as np
API_URL = "http://localhost:8000/generate_interaction"

db = load_db()

def process_image(frame, name, relationship, notes):
    name_unknown = name
    
    
    try:
            
        nparr = np.frombuffer(frame, np.uint8)

        if nparr.size == 0:
            return {"error": "Empty image"}

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Error: Unable to decode image", flush=True)
            return {"error": "Invalid image format"}

        face = extract_face(frame)
        embedding = get_embedding(face)

        # 2. Match
        matched_name, score = find_match(embedding, db)

        # ----------------------------
        # CASE 1: KNOWN PERSON
        # ----------------------------
        if matched_name and score > 8.75:
            print(f"[MATCH] {matched_name} ({score:.2f})", flush=True)

            person = db[matched_name]
            person["last_met"] = datetime.now().isoformat()
            save_db(db)

            event = {
                "person_id": person.get("id"),
                "name": matched_name,
                "relationship": person.get("relationship", "Unknown"),
                "last_met_timestamp": person.get("last_met"),
                "notes": person.get("notes", ""),
                "confidence_score": float(score),
                "is_unknown": False,
                "preferred_language": "English"
            }

        # ----------------------------
        # CASE 2: UNKNOWN PERSON
        # ----------------------------
        else:
            print("[NEW FACE]", flush=True)
            print("DEBUG INPUT IN PIPELINE:", name_unknown, relationship, notes, flush=True)

            if not name_unknown:
                return {
                    "error": "Unknown face detected. Send name to register."
                }

            person_id = str(uuid.uuid4())

            db[name_unknown] = {
                "id": person_id,
                "embedding": embedding,
                "relationship": relationship or "Unknown",
                "notes": notes or "",
                "last_met": datetime.now().isoformat()
            }

            save_db(db)
            print(f"[REGISTERED] {name} / ID: {person_id} / Relationship: {relationship} / Notes: {notes}", flush=True)

            return {
                "person_id": person_id,
                "name": name,
                "relationship": relationship,
                "last_met_timestamp": db[name]["last_met"],
                "notes": notes,
                "confidence_score": 0.0,
                "is_unknown": True,
                "preferred_language": "English"
            }

        response = requests.post(API_URL, json=event)

        if response.status_code == 200:
            data = response.json()
            print("[MITRA RESPONSE]", data, flush=True)

            if data.get("audio_url"):
                print("Audio URL:", data["audio_url"], flush=True)
        else:
            print("API Error:", response.text)

    except Exception as e:
        print("Error:", e)

    return {
    "error": "Unhandled pipeline case"
}