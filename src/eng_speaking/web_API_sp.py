# web_API_sp.py  ← FINAL ACCESSIBLE WEB APP
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64

from yolo_detector_API_sp import detect_and_get_result

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessible Object Detector</title>
    <style>
        body {font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1e3a8a, #3b82f6); margin:0; padding:20px; min-height:100vh; color:#333;}
        .container {max-width: 1000px; margin: 40px auto; background: white; border-radius: 24px; overflow: hidden; box-shadow: 0 25px 50px rgba(0,0,0,0.3);}
        header {background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 50px; text-align: center;}
        h1 {margin: 0; font-size: 3.2em; font-weight: 900;}
        .subtitle {font-size: 1.4em; opacity: 0.9; margin-top: 10px;}
        .content {padding: 50px;}
        .upload-box {border: 5px dashed #93c5fd; padding: 70px; text-align: center; border-radius: 20px; background: #f0f9ff; cursor: pointer; transition: 0.4s;}
        .upload-box:hover {background: #dbeafe; transform: scale(1.02);}
        .file-info {margin-top: 20px; font-size: 1.3em; font-weight: bold; color: #1d4ed8;}
        button {padding: 18px 50px; font-size: 1.4em; border-radius: 16px; cursor: pointer; transition: 0.3s;}
        #detectBtn {background: #94a3b8; color: white; border: none;}
        #detectBtn.active {background: #3b82f6;}
        #detectBtn.active:hover {background: #2563eb;}
        .result {margin-top: 50px; padding: 40px; background: #f0fdf4; border-radius: 20px; border-left: 10px solid #10b981; text-align: center;}
        .summary {font-size: 2em; line-height: 1.6; color: #166534; font-weight: bold;}
        .speak-btn {background: #f59e0b; color: white; border: none; padding: 20px 40px; font-size: 1.5em; border-radius: 16px; margin: 20px;}
        .speak-btn:hover {background: #d97706;}
        .result-img {max-width: 100%; border-radius: 16px; box-shadow: 0 20px 40px rgba(0,0,0,0.2); margin: 30px 0;}
        .download {background: #10b981; color: white; padding: 16px 40px; text-decoration: none; border-radius: 12px; font-size: 1.2em;}
        .download:hover {background: #059669;}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Accessible Object Detector</h1>
            <p class="subtitle">For Everyone — Including Blind & Visually Impaired Users</p>
        </header>
        <div class="content">
            <form action="/" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="upload-box" id="dropZone" onclick="document.getElementById('file').click()">
                    <p style="font-size:2em; margin:15px 0;">Click or Drop Image Here</p>
                    <p>Supports: JPG, PNG, BMP</p>
                    <div class="file-info" id="fileInfo"></div>
                    <input type="file" id="file" name="file" accept="image/*" required style="display:none">
                </div>
                <center>
                    <button type="submit" id="detectBtn" disabled>Detect Objects</button>
                </center>
            </form>

            <div id="resultSection">
                <!-- RESULT WILL BE INSERTED HERE -->
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file');
        const fileInfo = document.getElementById('fileInfo');
        const detectBtn = document.getElementById('detectBtn');

        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) {
                fileInfo.textContent = "Image selected: " + fileInput.files[0].name;
                detectBtn.disabled = false;
                detectBtn.classList.add('active');
            }
        });

        document.getElementById('uploadForm').onsubmit = () => {
            detectBtn.disabled = true;
            detectBtn.textContent = "Analyzing... Please wait";
        };

        function speakText(text) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            utterance.rate = 0.9;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTML_PAGE.replace("<!-- RESULT WILL BE INSERTED HERE -->", "")

@app.post("/", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        result_html = '<div class="result"><h2>Error</h2><p>Cannot read image!</p></div>'
    else:
        result_img, speech_text = detect_and_get_result(img)
        _, buffer = cv2.imencode(".jpg", result_img)
        img_b64 = base64.b64encode(buffer).decode()

        result_html = f'''
        <div class="result">
            <h2>Detection Complete</h2>
            <div class="summary" id="speechText">{speech_text}</div>
            <button onclick="speakText(document.getElementById('speechText').innerText)" class="speak-btn">
                Listen to Result
            </button>
            <img src="data:image/jpeg;base64,{img_b64}" class="result-img" alt="Detection result">
            <br>
            <a href="data:image/jpeg;base64,{img_b64}" download="accessible_detection_result.jpg" class="download">
                Download Image
            </a>
        </div>
        '''

    return HTML_PAGE.replace("<!-- RESULT WILL BE INSERTED HERE -->", result_html)

if __name__ == "__main__":
    print("═" * 80)
    print(" YOLOv11n FULLY ACCESSIBLE WEB DETECTOR")
    print(" → http://127.0.0.1:8000")
    print(" → Includes SPEAK BUTTON for blind users!")
    print("═" * 80)
    uvicorn.run("web_API_sp:app", host="127.0.0.1", port=8000, log_level="critical")