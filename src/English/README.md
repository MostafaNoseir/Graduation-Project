# Face Recognition System

## Introduction
This project allows registering and recognizing people using only face images. It uses the MobileFaceNet.tflite model to extract face embeddings and compare them.

## Folders and Files
- `models/MobileFaceNet.tflite` : TFLite model for face recognition.
- `extract_embedding_en.py` : Extract embeddings from images.
- `enroll_person_en.py` : Register a new person or update their data in the database.
- `recognize_image_en.py` : Recognize people using a single image.
- `README.md` : This file.

## How to Enroll People
1. Prepare at least 5 images per person.
2. Run the command:
```bash
python enroll_person_en.py <person_name> <image1.jpg> <image2.jpg> ... <image5.jpg>
```
3. Embeddings will be calculated and saved in the database face_db.pkl.

## Recognize People
To recognize a person from a single image:
```bash
python recognize_image_en.py <image_path>
```
The program will compare the image with the registered embeddings and print the name if matched, or "Unknown person" if no match is found.

## Notes:
- Use clear face images without occlusions.
- The matching threshold can be adjusted inside recognize_image_en.py.
