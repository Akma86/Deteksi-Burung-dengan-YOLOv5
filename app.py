import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load model YOLOv5 menggunakan Torch Hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') #Diubah sesuai alamat file model

# UI Aplikasi Streamlit
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan")

# Memilih mode dengan radio button
mode = st.sidebar.radio("Pilih mode:", ["Live Webcam", "Upload Video", "Upload Gambar"])

# Fungsi pembantu untuk mendeteksi objek pada gambar
def detect_objects(img):
    # Melakukan deteksi objek dengan model
    results = model([img])
    detected_img = img.copy()
    
    # Menambahkan bounding box dan label pada gambar
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # Batas tingkat confidence
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 2)
    return detected_img

if mode == "Live Webcam":
    st.write("Mode Webcam Langsung")
    FRAME_WINDOW = st.image([])

    # Membuka koneksi ke webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Webcam tidak ditemukan atau tidak dapat diakses.")
    else:
        # Menampilkan video dari webcam secara real-time
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

    # Menutup koneksi webcam
    cap.release()

elif mode == "Upload Video":
    st.write("Mode Upload Video")
    # Mengunggah file video
    uploaded_video = st.file_uploader("Unggah file video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Menyimpan file video sementara
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        FRAME_WINDOW = st.image([])

        # Menampilkan video yang diunggah dan mendeteksi objek
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

        cap.release()

elif mode == "Upload Gambar":  # Sesuaikan dengan radio button
    st.write("Mode Upload Gambar")
    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        img = np.array(img)  # Convert to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Ubah ke format OpenCV
        
        detected_image = detect_objects(img)
        
        st.image(detected_image, caption="Objek yang Terdeteksi")




# Upload gambar

