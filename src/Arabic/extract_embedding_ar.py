# extract_embedding.py

import numpy as np
import cv2
import tensorflow as tf

class FaceEmbedder:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, img):
        """
        أفضل preprocessing للـ MobileFaceNet:
        - face crop & alignment
        - RGB
        - resize 112x112
        - normalize
        - batch dimension
        """
        # لو grayscale -> BGR
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simple face detection + crop (باستخدام OpenCV Haarcascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            # لو لم يتم الكشف، استخدم الصورة كاملة
            face_img = img
        else:
            # خذ أول وجه فقط
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]

        # resize
        face_img = cv2.resize(face_img, (112, 112))
        # normalize
        face_img = (face_img - 127.5) / 128.0
        face_img = face_img.astype(np.float32)
        # add batch
        inp = np.expand_dims(face_img, axis=0)
        return inp

    def get_embedding(self, img_or_path):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
            if img is None:
                print(f"خطأ: الصورة غير موجودة أو غير قابلة للفتح: {img_or_path}")
                return None
        else:
            img = img_or_path

        inp = self.preprocess(img)  # batch=1

        # solve batch mismatch
        expected_batch = self.input_details[0]['shape'][0]
        if inp.shape[0] != expected_batch:
            inp = np.concatenate([inp]*expected_batch, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]['index'])  # (batch,512)
        return emb[0]  # أول embedding فقط
