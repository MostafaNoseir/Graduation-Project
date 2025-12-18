# enroll_person_en.py
import sys
import os
import numpy as np
import pickle
from extract_embedding_en import FaceEmbedder

DB_FILE = "face_db.pkl"

def load_db():
    if not os.path.exists(DB_FILE):
        return {"names": [], "embeddings": []}
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def enroll(name, image_paths):
    if not name:
        print("Name is empty")
        return
    if len(image_paths) < 5:
        print("Recommended to use at least 5 images per person")
        return

    embedder = FaceEmbedder("MobileFaceNet.tflite")
    embeddings = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        emb = embedder.get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) == 0:
        print("No embeddings were extracted.")
        return

    mean_emb = np.mean(embeddings, axis=0)

    db = load_db()
    if name in db["names"]:
        idx = db["names"].index(name)
        db["embeddings"][idx] = mean_emb
        print(f"Updated embeddings for '{name}' with {len(embeddings)} images.")
    else:
        db["names"].append(name)
        db["embeddings"].append(mean_emb)
        print(f"Registered '{name}' successfully with {len(embeddings)} images.")

    save_db(db)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python enroll_person_en.py <name> <image1> <image2> ... <image5>")
        sys.exit(1)

    name = sys.argv[1]
    images = sys.argv[2:]
    enroll(name, images)
