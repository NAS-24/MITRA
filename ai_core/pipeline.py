import cv2
import time
import json

from vision import get_face, get_embedding
from matcher import match_embedding


# Load database
with open(r"F:\Mitra\MITRA\ai_core\data.json", "r") as f:
    database = json.load(f)


cap = cv2.VideoCapture(0)

last_embedding_time = 0
INTERVAL = 2  # seconds


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    face = get_face(frame)

    current_time = time.time()

    if face is not None and (current_time - last_embedding_time > INTERVAL):

        embedding = get_embedding(face)

        if embedding is not None:
            person_id, score = match_embedding(embedding, database)

            result = {
                "person_id": person_id,
                "confidence_score": round(score, 3),
                "is_unknown": person_id is None
            }

            print(result)

        last_embedding_time = current_time

    cv2.imshow("MITRA Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()