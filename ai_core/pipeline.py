import cv2
from matcher import load_db, save_db, get_embedding, find_match

db = load_db()

cap = cv2.VideoCapture(0)

print("Press C to capture | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        try:
            embedding = get_embedding(frame)
            name, score = find_match(embedding, db)

            if name:
                print(f"[MATCH] {name} ({score:.2f})")
            else:
                new_name = input("Enter name: ")
                db[new_name] = embedding
                save_db(db)
                print("[SAVED] New face added")

        except Exception as e:
            print("Error:", e)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()