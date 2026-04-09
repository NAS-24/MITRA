from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from matcher import load_db, get_embedding, find_match, build_response

app = FastAPI()
db = load_db()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert bytes → image
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        embedding = get_embedding(frame)
        name, score = find_match(embedding, db)

        response = build_response(name, score)
        return response

    except Exception as e:
        return {"error": str(e)}