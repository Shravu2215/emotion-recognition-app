# ==========================================================
# CLOUD-READY VERSION: Facial Expression Recognition
# Uses streamlit-webrtc so webcam is captured in the BROWSER
# (works both locally AND when deployed on a cloud server)
# ==========================================================
# Run with: streamlit run app.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import load_model

# ---- CONFIG ----
MODEL_PATH = "best_emotion_model.h5"
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
HISTORY_LENGTH = 30

st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("AI-Based Facial Expression Recognition & Emotional State Analysis")
st.caption("Deployed on AWS Cloud | Real-time browser webcam capture")

# ---- LOAD MODEL & FACE DETECTOR (cached) ----
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, face_cascade

model, face_cascade = load_resources()

# ---- VIDEO PROCESSOR ----
# This class runs in a separate thread that streamlit-webrtc manages.
# It receives each browser webcam frame, runs detection, and stores
# results in a thread-safe way so the main app can read them.
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.history = []
        self.current_emotion = "No face detected"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))

        detected_emotion = "No face detected"

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            detected_emotion = EMOTIONS[emotion_idx]
            confidence = float(np.max(prediction)) * 100

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{detected_emotion} ({confidence:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        with self.lock:
            self.current_emotion = detected_emotion
            self.history.append(detected_emotion)
            if len(self.history) > HISTORY_LENGTH:
                self.history.pop(0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---- RTC CONFIG ----
# STUN server helps the browser and server find each other over the internet.
# This is required for cloud deployment (not needed for localhost testing,
# but keeping it makes the code work in both cases).
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---- LAYOUT ----
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Current Emotion")
    emotion_placeholder = st.empty()
    st.subheader("Emotion Trend (recent readings)")
    chart_placeholder = st.empty()

    if ctx.video_processor:
        # Streamlit reruns this block periodically while the stream is active
        with ctx.video_processor.lock:
            current = ctx.video_processor.current_emotion
            history = list(ctx.video_processor.history)

        emotion_placeholder.markdown(f"## {current}")

        if history:
            counts = pd.Series(history).value_counts()
            counts = counts.reindex(EMOTIONS, fill_value=0)
            chart_placeholder.bar_chart(counts)
    else:
        st.info("Click 'START' above to begin detection.")
