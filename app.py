# ==========================================================
# CLOUD-READY VERSION: Facial Expression Recognition
# Uses streamlit-webrtc so webcam is captured in the BROWSER
# (works both locally AND when deployed on a cloud server)
# ==========================================================
# Run with: streamlit run app.py

import threading

import av
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
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
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.history = []
        self.current_emotion = "No face detected"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
        )

        detected_emotion = "No face detected"

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            detected_emotion = EMOTIONS[emotion_idx]
            confidence = float(np.max(prediction)) * 100

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{detected_emotion} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        with self.lock:
            self.current_emotion = detected_emotion
            self.history.append(detected_emotion)
            if len(self.history) > HISTORY_LENGTH:
                self.history.pop(0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---- FETCH TURN CREDENTIALS FROM METERED ----
# NOTE: @st.cache_resource intentionally NOT used here. If a fetch fails once,
# the STUN-only fallback must not get stuck. We cache only on SUCCESS,
# in st.session_state.
def get_ice_servers():
    if "ice_servers" in st.session_state:
        return st.session_state["ice_servers"]

    try:
        metered_domain = st.secrets["METERED_DOMAIN"].strip()
        secret_key = st.secrets["METERED_SECRET_KEY"].strip()

        # Step 1: create a credential using the secret key
        create_resp = requests.post(
            f"https://{metered_domain}/api/v1/turn/credential",
            params={"secretKey": secret_key},
            json={"expiryInSeconds": 3600, "label": "streamlit-app"},
            timeout=10,
        )
        create_data = create_resp.json()
        if "apiKey" not in create_data:
            raise ValueError(f"Could not create credential: {create_data}")

        # Step 2: use that credential's apiKey to fetch the real ICE servers
        get_resp = requests.get(
            f"https://{metered_domain}/api/v1/turn/credentials",
            params={"apiKey": create_data["apiKey"]},
            timeout=10,
        )
        ice_servers = get_resp.json()

        if not isinstance(ice_servers, list) or len(ice_servers) == 0:
            raise ValueError(f"Unexpected ICE servers format: {ice_servers}")

        st.session_state["ice_servers"] = ice_servers  # cache only on success
        return ice_servers

    except Exception as e:
        st.warning(f"Could not fetch TURN credentials, falling back to STUN only: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ice = get_ice_servers()

# Count how many TURN entries we actually got (0 = only STUN)
n_turn = 0
for s in ice:
    urls = s.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    if any(u.startswith("turn") for u in urls):
        n_turn += 1
st.caption(f"ICE servers: {len(ice)} (TURN entries: {n_turn})")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice})

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
