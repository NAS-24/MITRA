import numpy as np

THRESHOLD = 0.75  # prototype threshold


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) #cosθ = (a · b) / (||a|| × ||b||)


def match_embedding(embedding, database):
    best_score = -1
    best_id = None
    print(database.items())

    for person_id, embeddings in database.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)

            if score > best_score:
                best_score = score
                best_id = person_id

    if best_score < THRESHOLD:
        return None, float(best_score)

    return best_id, float(best_score)