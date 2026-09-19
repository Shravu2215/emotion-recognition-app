# ==========================================================
# Facial Expression Recognition + Emotional State Analysis
# streamlit-webrtc (browser webcam) + Metered TURN
# Run with: streamlit run app.py
# ==========================================================

import threading
import time
from collections import Counter, deque

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
SMOOTH_WINDOW = 10        # majority vote over last N predictions
RECORD_EVERY_SEC = 0.2    # log at most 5 readings per second
RECENT_LENGTH = 30

# Mood weights: +1 = very positive, -1 = very negative
MOOD_WEIGHTS = {
    "Happy": 1.0, "Surprise": 0.5, "Neutral": 0.0,
    "Fear": -0.5, "Sad": -0.75, "Angry": -1.0, "Disgust": -1.0,
}
POSITIVE = ["Happy", "Surprise"]
NEGATIVE = ["Angry", "Disgust", "Fear", "Sad"]

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
        self.recent = []                       # last few smoothed labels
        self.records = []                      # full session log: (seconds, emotion)
        self.current_emotion = "No face detected"
        self._window = deque(maxlen=SMOOTH_WINDOW)
        self._start = time.time()
        self._last_record = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
        )

        label = "No face detected"

        if len(faces) > 0:
            # use the largest face only
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=(0, -1))

            prediction = model.predict(face_roi, verbose=0)
            raw_emotion = EMOTIONS[int(np.argmax(prediction))]
            confidence = float(np.max(prediction)) * 100

            # smoothing: majority vote of last N predictions
            self._window.append(raw_emotion)
            label = Counter(self._window).most_common(1)[0][0]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({confidence:.0f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.time()
            with self.lock:
                if now - self._last_record >= RECORD_EVERY_SEC:
                    self.records.append((round(now - self._start, 1), label))
                    self._last_record = now
                self.recent.append(label)
                if len(self.recent) > RECENT_LENGTH:
                    self.recent.pop(0)
        else:
            self._window.clear()

        with self.lock:
            self.current_emotion = label

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---- FETCH TURN CREDENTIALS FROM METERED ----
# Cached only on SUCCESS (in session_state), so a failed fetch never sticks.
def get_ice_servers():
    if "ice_servers" in st.session_state:
        return st.session_state["ice_servers"]
    try:
        domain = st.secrets["METERED_DOMAIN"].strip()
        secret_key = st.secrets["METERED_SECRET_KEY"].strip()

        create_resp = requests.post(
            f"https://{domain}/api/v1/turn/credential",
            params={"secretKey": secret_key},
            json={"expiryInSeconds": 3600, "label": "streamlit-app"},
            timeout=10,
        )
        create_data = create_resp.json()
        if "apiKey" not in create_data:
            raise ValueError(f"Could not create credential: {create_data}")

        get_resp = requests.get(
            f"https://{domain}/api/v1/turn/credentials",
            params={"apiKey": create_data["apiKey"]},
            timeout=10,
        )
        servers = get_resp.json()
        if not isinstance(servers, list) or len(servers) == 0:
            raise ValueError(f"Unexpected ICE servers format: {servers}")

        st.session_state["ice_servers"] = servers
        return servers
    except Exception as e:
        st.warning(f"Could not fetch TURN credentials, falling back to STUN only: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ice = get_ice_servers()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice})


# ---- ANALYSIS HELPERS ----
def build_dataframe(records):
    df = pd.DataFrame(records, columns=["time_sec", "emotion"])
    df["mood_value"] = df["emotion"].map(MOOD_WEIGHTS)
    return df


def mood_score(df):
    """0-100 score: 50 = neutral overall, 100 = fully positive, 0 = fully negative."""
    return (df["mood_value"].mean() + 1) / 2 * 100


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
    st.subheader("Recent Readings")
    chart_placeholder = st.empty()
    st.subheader("Live Mood Score")
    score_placeholder = st.empty()

# ---- LIVE UPDATE LOOP (runs while the stream is playing) ----
if ctx.state.playing:
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is not None:
            with proc.lock:
                current = proc.current_emotion
                recent = list(proc.recent)
                records = list(proc.records)

            st.session_state["records"] = records  # keep data after STOP

            emotion_placeholder.markdown(f"## {current}")
            if recent:
                counts = pd.Series(recent).value_counts().reindex(EMOTIONS, fill_value=0)
                chart_placeholder.bar_chart(counts)
            if records:
                score_placeholder.markdown(
                    f"## {mood_score(build_dataframe(records)):.0f} / 100"
                )
        time.sleep(1)
else:
    emotion_placeholder.markdown("## -")
    st.info("Click 'START' above to begin detection.")

# ---- SESSION SUMMARY (shown after the stream is stopped) ----
records = st.session_state.get("records", [])
if records and not ctx.state.playing:
    df = build_dataframe(records)
    total = len(df)

    st.divider()
    st.header("Session Summary")

    pos_pct = df["emotion"].isin(POSITIVE).sum() / total * 100
    neg_pct = df["emotion"].isin(NEGATIVE).sum() / total * 100
    neu_pct = 100 - pos_pct - neg_pct
    dominant = df["emotion"].value_counts().idxmax()
    duration = df["time_sec"].max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mood Score", f"{mood_score(df):.0f} / 100")
    m2.metric("Dominant Emotion", dominant)
    m3.metric("Positive / Neutral / Negative", f"{pos_pct:.0f}% / {neu_pct:.0f}% / {neg_pct:.0f}%")
    m4.metric("Session Length", f"{duration:.0f} s")

    left, right = st.columns(2)
    with left:
        st.subheader("Emotion Distribution")
        dist = (df["emotion"].value_counts() / total * 100).reindex(EMOTIONS, fill_value=0)
        st.bar_chart(dist)
    with right:
        st.subheader("Mood Timeline")
        timeline = df.set_index("time_sec")["mood_value"].rolling(10, min_periods=1).mean()
        st.line_chart(timeline)
        st.caption("Above 0 = positive mood, below 0 = negative mood (smoothed).")

    st.download_button(
        "Download session data (CSV)",
        data=df[["time_sec", "emotion"]].to_csv(index=False).encode("utf-8"),
        file_name="emotion_session.csv",
        mime="text/csv",
    )

    st.caption(
        "Mood score = average of emotion weights "
        "(Happy +1, Surprise +0.5, Neutral 0, Fear -0.5, Sad -0.75, Angry/Disgust -1), scaled to 0-100."
    )
