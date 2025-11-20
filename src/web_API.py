# web_API.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64

from yolo_detector_API import detect_and_draw_only

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv11n Real-Time Detector</title>
    <style>
        body {font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin:0; padding:20px; min-height:100vh;}
        .container {max-width: 1000px; margin: 40px auto; background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.3);}
        header {background: #3b82f6; color: white; padding: 40px; text-align: center;}
        h1 {margin: 0; font-size: 3em;}
        .content {padding: 40px;}
        .upload-box {border: 4px dashed #93c5fd; padding: 60px; text-align: center; border-radius: 16px; background: #f8fafc; cursor: pointer; transition: 0.3s;}
        .upload-box:hover {background: #e0f2fe;}
        .upload-box.active {border-color: #3b82f6; background: #dbeafe;}
        .file-info {margin-top: 15px; font-weight: bold; color: #1d4ed8;}
        button {background: #94a3b8; color: white; border: none; padding: 16px 40px; font-size: 1.3em; border-radius: 12px; cursor: not-allowed; margin-top: 20px; transition: 0.3s;}
        button.active {background: #3b82f6; cursor: pointer;}
        button.active:hover {background: #2563eb;}
        button:disabled {background: #94a3b8; cursor: not-allowed;}
        .result {margin-top: 40px; padding: 30px; background: #f0fdf4; border-radius: 16px; text-align: center; border-left: 8px solid #10b981;}
        .summary {font-size: 1.6em; font-weight: bold; color: #166534; white-space: pre-line; margin: 20px 0;}
        .result-img {max-width: 100%; border-radius: 16px; box-shadow: 0 15px 35px rgba(0,0,0,0.2); margin: 20px 0;}
        .download {background: #10b981; padding: 14px 32px; margin: 15px; text-decoration: none; color: white; border-radius: 10px; display: inline-block;}
        .download:hover {background: #059669;}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>YOLOv11n Detector</h1>
            <p>Upload Image → Click Detect → Instant Result</p>
        </header>
        <div class="content">
            <form action="/" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="upload-box" id="dropZone" onclick="document.getElementById('file').click()">
                    <p style="font-size:1.5em; margin:10px 0;">Click or Drop Image Here</p>
                    <p>JPG, PNG, BMP</p>
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
        const dropZone = document.getElementById('dropZone');
        const detectBtn = document.getElementById('detectBtn');
        const resultSection = document.getElementById('resultSection');

        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                const fileName = fileInput.files[0].name;
                fileInfo.textContent = "Image selected: " + fileName;
                dropZone.classList.add('active');
                detectBtn.disabled = false;
                detectBtn.classList.add('active');
                detectBtn.textContent = "Detect Objects";
            }
        });

        // Drag & drop support
        ['dragover', 'dragenter'].forEach(evt => {
            dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('active'); });
        });
        ['dragleave', 'drop'].forEach(evt => {
            dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('active'); });
        });

        // Form submit
        document.getElementById('uploadForm').onsubmit = function() {
            detectBtn.disabled = true;
            detectBtn.textContent = "Detecting...";
        };
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
        result_html = '<div class="result"><h2>Error</h2><p>Could not read the image!</p></div>'
    else:
        result_img, summary = detect_and_draw_only(img)
        _, buffer = cv2.imencode(".jpg", result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_b64 = base64.b64encode(buffer).decode()

        result_html = f'''
        <div class="result">
            <h2>Detection Result</h2>
            <div class="summary">{summary.replace(chr(10), "<br>")}</div>
            <img src="data:image/jpeg;base64,{img_b64}" class="result-img">
            <br>
            <a href="data:image/jpeg;base64,{img_b64}" download="yolov11_result.jpg" class="download">
                Download Result Image
            </a>
        </div>
        '''

    return HTML_PAGE.replace("<!-- RESULT WILL BE INSERTED HERE -->", result_html)

if __name__ == "__main__":
    print("\n" + "═" * 75)
    print(" YOLOv11n REAL-TIME WEB DETECTOR — FINAL PERFECT VERSION")
    print(" Open → http://127.0.0.1:8000")
    print("   → Click box → Choose image → Button turns blue → Click Detect!")
    print("═" * 75)
    uvicorn.run("web_API:app", host="127.0.0.1", port=8000, log_level="critical")