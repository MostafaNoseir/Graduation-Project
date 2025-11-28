# web_arabic.py  ← النسخة النهائية مع زر التنزيل
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64

from yolo_arabic import detect_arabic

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>كاشف الأشياء الذكي YOLOv11n</title>
    <style>
        body {font-family: 'Cairo', 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0f766e, #14b8a6); margin:0; padding:20px;}
        .container {max-width: 1100px; margin: 40px auto; background: white; border-radius: 28px; overflow: hidden; box-shadow: 0 30px 70px rgba(0,0,0,0.4);}
        header {background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 60px; text-align: center;}
        h1 {margin: 0; font-size: 3.8em; font-weight: 900;}
        .content {padding: 60px;}
        .upload-box {border: 6px dashed #99f6e4; padding: 80px; text-align: center; border-radius: 24px; background: #f0fdfa; cursor: pointer; transition: 0.4s;}
        .upload-box:hover {background: #ccfbf1; transform: translateY(-5px);}
        button {padding: 22px 70px; font-size: 1.8em; border-radius: 20px; margin: 25px;}
        #detectBtn {background: #94a3b8; color: white; border: none;}
        #detectBtn.active {background: #06b6d4;}
        .result {margin-top: 60px; padding: 50px; background: #f0fdfa; border-radius: 24px; border-right: 14px solid #10b981; text-align: center;}
        .summary {font-size: 2.5em; line-height: 2; color: #166534; font-weight: bold; margin: 30px 0;}
        .speak-btn {background: #ea580c; color: white; border: none; padding: 28px 60px; font-size: 2em; border-radius: 24px;}
        .speak-btn:hover {background: #c2410c;}
        .download-btn {background: #0891b2; color: white; padding: 20px 50px; font-size: 1.6em; border-radius: 16px; text-decoration: none; display: inline-block; margin: 20px;}
        .download-btn:hover {background: #0e7490;}
        .result-img {max-width: 100%; border-radius: 20px; box-shadow: 0 25px 50px rgba(0,0,0,0.3); margin: 40px 0;}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@700;900&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header>
            <h1>كاشف الأشياء الذكي</h1>
            <p style="font-size:1.6em; opacity:0.95;">ارفع الصورة ← اضغط كشف ← استمع ونزّل النتيجة</p>
        </header>
        <div class="content">
            <form action="/" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="upload-box" id="dropZone" onclick="document.getElementById('file').click()">
                    <p style="font-size:2.5em;">اضغط أو اسحب الصورة هنا</p>
                    <p style="font-size:1.4em;">يدعم: JPG, PNG, BMP</p>
                    <div id="fileInfo" style="font-size:1.6em; margin-top:20px; color:#0d9488; font-weight:bold;"></div>
                    <input type="file" id="file" name="file" accept="image/*" required style="display:none">
                </div>
                <center>
                    <button type="submit" id="detectBtn" disabled>ابدأ الكشف الآن</button>
                </center>
            </form>

            <div id="resultSection"></div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file');
        const fileInfo = document.getElementById('fileInfo');
        const detectBtn = document.getElementById('detectBtn');

        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) {
                fileInfo.textContent = "تم اختيار: " + fileInput.files[0].name;
                detectBtn.disabled = false;
                detectBtn.classList.add('active');
            }
        });

        document.getElementById('uploadForm').onsubmit = () => {
            detectBtn.disabled = true;
            detectBtn.textContent = "جاري التحليل... انتظر قليلاً";
        };

        function playArabic(text) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'ar-SA';
            utterance.rate = 0.9;
            speechSynthesis.speak(utterance);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTML_PAGE

@app.post("/", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        result_html = '<div class="result"><h2>خطأ</h2><p>تعذر قراءة الصورة!</p></div>'
    else:
        result_img, arabic_text = detect_arabic(img)
        _, buffer = cv2.imencode(".jpg", result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_b64 = base64.b64encode(buffer).decode()

        result_html = f'''
        <div class="result">
            <h2>تم الكشف بنجاح!</h2>
            <div class="summary" id="arabicResult">{arabic_text}</div>
            <button onclick="playArabic(document.getElementById('arabicResult').innerText)" class="speak-btn">
                استمع للنتيجة
            </button>
            <img src="data:image/jpeg;base64,{img_b64}" class="result-img" alt="النتيجة مع الصناديق">
            <br>
            <a href="data:image/jpeg;base64,{img_b64}" download="نتيجة_الكشف.jpg" class="download-btn">
                تنزيل الصورة الناتجة
            </a>
        </div>
        '''

    return HTML_PAGE.replace('<div id="resultSection"></div>', f'<div id="resultSection">{result_html}</div>')

if __name__ == "__main__":
    print("═" * 85)
    print(" كاشف الأشياء YOLOv11n - النسخة العربية النهائية المثالية")
    print(" افتح المتصفح: http://127.0.0.1:8000")
    print("═" * 85)
    uvicorn.run("web_arabic:app", host="127.0.0.1", port=8000, log_level="critical")