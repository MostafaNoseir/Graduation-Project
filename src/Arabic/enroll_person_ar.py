# enroll_person.py

import sys
import os
import numpy as np
import pickle
from extract_embedding import FaceEmbedder

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
        print("الاسم فارغ")
        return
    if len(image_paths) < 5:
        print("ينصح باستخدام 5 صور على الأقل لكل شخص")
        return

    embedder = FaceEmbedder("MobileFaceNet.tflite")
    embeddings = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"الصورة غير موجودة: {img_path}")
            continue
        emb = embedder.get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) == 0:
        print("لم يتم استخراج أي embeddings.")
        return

    mean_emb = np.mean(embeddings, axis=0)

    db = load_db()
    if name in db["names"]:
        idx = db["names"].index(name)
        db["embeddings"][idx] = mean_emb
        print(f"تم تحديث embeddings للشخص '{name}' مع {len(embeddings)} صور.")
    else:
        db["names"].append(name)
        db["embeddings"].append(mean_emb)
        print(f"تم تسجيل الشخص '{name}' بنجاح مع {len(embeddings)} صور.")

    save_db(db)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("استخدام: python enroll_person.py <اسم> <صورة1> <صورة2> ... <صورة5>")
        sys.exit(1)

    name = sys.argv[1]
    images = sys.argv[2:]
    enroll(name, images)
