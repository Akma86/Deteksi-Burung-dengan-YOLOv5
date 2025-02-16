import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pygame
import time

# Inisialisasi pygame untuk suara
pygame.mixer.init()
sound_path = "Siren.mp3"  # Ganti dengan path suara kamu
alert_sound = pygame.mixer.Sound(sound_path)

# Load model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# UI Streamlit
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan")

# Pilihan mode
mode = st.sidebar.radio("Pilih mode:", ["Live Webcam", "Upload Video", "Upload Gambar"])

# Fungsi deteksi objek
def detect_objects(img):
    results = model([img])
    detected_img = img.copy()
    detected = False  # Apakah ada objek yang terdeteksi?

    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:
            detected = True  # Ada objek yang terdeteksi
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return detected_img, detected

if mode == "Live Webcam":
    st.write("Mode Webcam Langsung")
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Webcam tidak ditemukan atau tidak dapat diakses.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame, detected = detect_objects(frame)

            if detected:
                alert_sound.play(-1)  # Loop suara
                st.markdown("<style>body {background-color: red;}</style>", unsafe_allow_html=True)
            else:
                alert_sound.stop()
                st.markdown("<style>body {background-color: white;}</style>", unsafe_allow_html=True)

            FRAME_WINDOW.image(detected_frame)

    cap.release()

elif mode == "Upload Video":
    st.write("Mode Upload Video")
    uploaded_video = st.file_uploader("Unggah file video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame, detected = detect_objects(frame)

            if detected:
                alert_sound.play(-1)
                st.markdown("<style>body {background-color: red;}</style>", unsafe_allow_html=True)
            else:
                alert_sound.stop()
                st.markdown("<style>body {background-color: white;}</style>", unsafe_allow_html=True)

            FRAME_WINDOW.image(detected_frame)

        cap.release()

elif mode == "Upload Gambar":
    st.write("Mode Upload Gambar")
    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        img = np.array(img)
        
        detected_image, detected = detect_objects(img)
        
        if detected:
            alert_sound.play()  # Mainkan suara sekali
            st.markdown("<style>body {background-color: red;}</style>", unsafe_allow_html=True)
            time.sleep(1)  # Tahan warna merah sebentar
            st.markdown("<style>body {background-color: white;}</style>", unsafe_allow_html=True)

        st.image(detected_image, caption="Objek yang Terdeteksi")
