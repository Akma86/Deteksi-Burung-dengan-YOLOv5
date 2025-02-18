import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model YOLOv5
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

model = load_model()

# Fungsi untuk mendeteksi objek dalam gambar
def detect_objects(img):
    results = model([img])
    detected_img = img.copy()

    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return detected_img

# Kelas untuk menangani streaming video
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        detected_img = detect_objects(img)
        return detected_img

# UI Streamlit
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan")

mode = st.sidebar.radio("Pilih mode:", ["Live Webcam", "Upload Video", "Upload Gambar"])

# Mode Live Webcam
if mode == "Live Webcam":
    st.write("Mode Live Webcam (WebRTC)")
    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

# Mode Upload Video
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
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

        cap.release()

# Mode Upload Gambar
elif mode == "Upload Gambar":
    st.write("Mode Upload Gambar")
    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        detected_image = detect_objects(img)

        st.image(detected_image, caption="Objek yang Terdeteksi", use_column_width=True)
