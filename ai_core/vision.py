import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def get_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        face = frame[y:y + h_box, x:x + w_box]

        return face

    return None

def get_embedding(face):
    if face is None or face.size == 0:
        return None

    try:
        # Convert to grayscale
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize
        face = cv2.resize(face, (32, 32))

    except:
        return None

    # Flatten
    embedding = face.flatten().astype(float)

    # Normalize
    embedding = embedding / 255.0

    return embedding.tolist()