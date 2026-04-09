import cv2
from matcher import load_db, save_db, get_embedding, find_match
from datetime import datetime
import uuid
import requests
from matcher import extract_face
API_URL = "http://localhost:8000/generate_interaction"

db = load_db()

cap = cv2.VideoCapture(0)

print("Press C to capture | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        try:
            # 1. Extract face + embedding
            face = extract_face(frame)
            embedding = get_embedding(face)

            # 2. Match
            name, score = find_match(embedding, db)

            # ----------------------------
            # CASE 1: KNOWN PERSON
            # ----------------------------
            if name:
                print(f"[MATCH] {name} ({score:.2f})")

                person = db[name]
                person["last_met"] = datetime.now().isoformat()
                save_db(db)

                event = {
                    "person_id": person.get("id"),
                    "name": name,
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
                print("[NEW FACE]")

                new_name = input("Enter name: ").strip()
                relationship = input("Relationship: ").strip()
                notes = input("Notes: ").strip()

                db[new_name] = {
                    "id": str(uuid.uuid4()),
                    "embedding": embedding,
                    "relationship": relationship,
                    "notes": notes,
                    "last_met": datetime.now().isoformat()
                }

                save_db(db)

                event = {
                    "person_id": db[new_name]["id"],
                    "name": new_name,
                    "relationship": relationship,
                    "last_met_timestamp": db[new_name]["last_met"],
                    "notes": notes,
                    "confidence_score": 0.0,
                    "is_unknown": True,
                    "preferred_language": "English"
                }

            # ----------------------------
            # SEND TO MITRA API
            # ----------------------------
            print("[SENDING]", event)

            response = requests.post(API_URL, json=event)

            if response.status_code == 200:
                data = response.json()
                print("[MITRA RESPONSE]", data)

                if data.get("audio_url"):
                    print("Audio URL:", data["audio_url"])
            else:
                print("API Error:", response.text)

        except Exception as e:
            print("Error:", e)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()