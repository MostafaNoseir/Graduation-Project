# recognize_image.py

import sys
import os
import numpy as np
import pickle
import cv2
from extract_embedding import FaceEmbedder

DB_FILE = "face_db.pkl"
THRESHOLD = 0.75  # يمكن تعديلها لاحقاً بعد اختبار الصور

def load_db():
    if not os.path.exists(DB_FILE):
        return {"names": [], "embeddings": []}
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def recognize(image_path):
    db = load_db()
    if len(db["names"]) == 0:
        print("قاعدة البيانات فارغة، لم يتم تسجيل أي شخص بعد.")
        return

    embedder = FaceEmbedder("MobileFaceNet.tflite")

    img = cv2.imread(image_path)
    if img is None:
        print(f"الصورة غير موجودة: {image_path}")
        return
    emb = embedder.get_embedding(img)

    best_score = -1
    best_name = None
    for name, db_emb in zip(db["names"], db["embeddings"]):
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= THRESHOLD:
        print(f"تم التعرف على الشخص: {best_name} (score={best_score:.2f})")
    else:
        print(f"الشخص غير معروف (أفضل تطابق: {best_name}, score={best_score:.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("استخدام: python recognize_image.py <مسار الصورة>")
        sys.exit(1)
    image_path = sys.argv[1]
    recognize(image_path)
