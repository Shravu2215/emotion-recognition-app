import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmotionAI Vision System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_PATH = "best_emotion_model.h5"
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SMOOTH_WINDOW = 10
RECORD_EVERY_SEC = 0.2

EMOTION_COLORS = {
    "Happy": "#EAB308",
    "Sad": "#3B82F6",
    "Angry": "#EF4444",
    "Fear": "#A855F7",
    "Disgust": "#22C55E",
    "Surprise": "#F97316",
    "Neutral": "#6B7280",
    "No face detected": "#9CA3AF"
}

EMOTION_ICONS = {
    "Happy": "😊", "Sad": "😢", "Angry": "😠", "Fear": "😨",
    "Disgust": "🤢", "Surprise": "😲", "Neutral": "😐", "No face detected": "👤"
}

# ============================================================
# THEME & STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F1F5F9;
        color: #0F172A;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94A3B8;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }
    
    /* Style the radio buttons in the sidebar */
    div.row-widget.stRadio > div {
        background-color: transparent;
    }
    div.row-widget.stRadio > div > label {
        color: #F8FAFC !important;
        font-weight: 500;
        padding: 10px 14px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    /* Top Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border: 1px solid #E2E8F0;
    }
    .header-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #ECFDF5;
        color: #059669;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #D1FAE5;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
    }
    
    /* Cards */
    .ai-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px -1px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Metrics Row */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-bottom: 16px;
    }
    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: left;
    }
    .stat-label {
        font-size: 11px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .stat-val {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Probabilities Bar */
    .prob-container {
        margin-bottom: 12px;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 500;
        color: #334155;
        margin-bottom: 4px;
    }
    .prob-track {
        width: 100%;
        background-color: #F1F5F9;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        color: #64748B;
        text-align: center;
    }
    .empty-state i {
        font-size: 40px;
        margin-bottom: 16px;
        color: #94A3B8;
    }
    .empty-state h3 {
        color: #334155 !important;
        font-size: 18px !important;
        margin: 0 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL (CACHED)
# ============================================================

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, cascade

try:
    model, face_cascade = load_resources()
except Exception as e:
    st.error("❌ AI model could not be loaded. Please ensure the model file is valid.")
    st.exception(e)
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = []
if "session_start" not in st.session_state:
    st.session_state.session_start = None

# ============================================================
# EMOTION PROCESSOR
# ============================================================

def analyze_frame(img_bgr):
    """
    Runs face detection + emotion prediction on a single BGR image (numpy array).
    Returns (annotated_img_bgr, label, confidence, probabilities, faces_count)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    label = "No face detected"
    confidence = 0
    probabilities = np.zeros(len(EMOTIONS))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        prediction = model.predict(face, verbose=0)[0]
        label = EMOTIONS[int(np.argmax(prediction))]
        confidence = float(np.max(prediction)) * 100
        probabilities = prediction * 100

        color_hex = EMOTION_COLORS.get(label, "#6366F1").lstrip('#')
        box_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))  # BGR
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_color, 2)
        cv2.rectangle(img_bgr, (x, y - 30), (x + w, y), box_color, -1)
        cv2.putText(img_bgr, f"{label} {confidence:.0f}%", (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_bgr, label, confidence, probabilities, len(faces)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 30px;">
            <div style="background: #4F46E5; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px;">
                👁️
            </div>
            <div style="font-size: 18px; font-weight: 700; color: white; letter-spacing: 0.02em;">EmotionAI</div>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation", 
        ["📷 Live Dashboard", "📈 Session Logs", "⚙️ Settings"],
        label_visibility="collapsed"
    )

# ============================================================
# MAIN DASHBOARD
# ============================================================

if page == "📷 Live Dashboard":

    # Header
    st.markdown("""
        <div class="header-container">
            <div class="header-title">
                <span>Vision Center</span>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div> System Ready
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([1.6, 1], gap="medium")

    with col_main:
        # Camera Section
        st.markdown("""
            <div class="card-title" style="margin-bottom: 8px;">
                📷 Detection Stream
            </div>
        """, unsafe_allow_html=True)
        
        captured = st.camera_input("Take a photo", label_visibility="collapsed")

        current = "No face detected"
        conf = 0
        probs = np.zeros(len(EMOTIONS))
        faces_cnt = 0

        if captured is not None:
            file_bytes = np.asarray(bytearray(captured.getvalue()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, current, conf, probs, faces_cnt = analyze_frame(img_bgr)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            if st.session_state.session_start is None:
                st.session_state.session_start = time.time()
            now = time.time()
            st.session_state.records.append(
                (round(now - st.session_state.session_start, 1), current, conf)
            )
        
        # Analytics Section (Below Camera)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Session Analytics</div>', unsafe_allow_html=True)
        
        if len(st.session_state.records) > 0:
            df = pd.DataFrame(st.session_state.records, columns=["time", "emotion", "confidence"])
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Emotion Distribution")
                dist = df["emotion"].value_counts()
                st.bar_chart(dist, height=200)
            with c2:
                st.caption("Confidence Timeline")
                st.line_chart(df.set_index("time")["confidence"], height=200)
        else:
            st.markdown("""
                <div style="text-align: center; color: #94A3B8; padding: 30px 0; font-size: 13px;">
                    Insufficient data. Start the camera to generate session analytics.
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Current Detection</div>', unsafe_allow_html=True)
        
        # Placeholders for dynamic data
        metrics_ph = st.empty()
        probs_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update panel based on the latest captured photo (if any)
        if captured is not None:
            mins, secs = 0, 0
            if st.session_state.session_start is not None:
                duration = int(time.time() - st.session_state.session_start)
                mins, secs = divmod(duration, 60)

            metrics_ph.markdown(
                '<div class="metric-grid">'
                f'<div class="stat-box"><div class="stat-label">Expression</div>'
                f'<div class="stat-val" style="font-size:16px;">{EMOTION_ICONS.get(current, "")} {current}</div></div>'
                f'<div class="stat-box"><div class="stat-label">Confidence</div>'
                f'<div class="stat-val">{conf:.0f}%</div></div>'
                f'<div class="stat-box"><div class="stat-label">Faces</div>'
                f'<div class="stat-val">{faces_cnt}</div></div>'
                f'<div class="stat-box"><div class="stat-label">Duration</div>'
                f'<div class="stat-val">{mins}:{secs:02d}</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

            probs_html = '<div style="margin-top: 24px;"><div class="card-title">Live Probabilities</div>'
            for i, emotion in enumerate(EMOTIONS):
                p = probs[i]
                color = EMOTION_COLORS[emotion]
                probs_html += (
                    f'<div class="prob-container">'
                    f'<div class="prob-label"><span>{emotion}</span><span>{p:.1f}%</span></div>'
                    f'<div class="prob-track">'
                    f'<div class="prob-fill" style="width: {p}%; background-color: {color};"></div>'
                    f'</div></div>'
                )
            probs_html += '</div>'
            probs_ph.markdown(probs_html, unsafe_allow_html=True)
        else:
            metrics_ph.markdown(f"""
                <div class="empty-state">
                    <i>📷</i>
                    <h3>Ready for analysis</h3>
                    <p style="font-size: 13px; margin: 0;">Click "Take Photo" above to begin facial expression detection.</p>
                </div>
            """, unsafe_allow_html=True)
            probs_ph.empty()


elif page == "📈 Session Logs":
    st.markdown("""
        <div class="header-container">
            <div class="header-title">
                <span>Session Logs</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    if len(st.session_state.records) > 0:
        df = pd.DataFrame(st.session_state.records, columns=["Time (s)", "Predicted Emotion", "Confidence (%)"])
        st.dataframe(df.tail(100), use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="emotion_history.csv", mime="text/csv")
    else:
        st.markdown("""
            <div class="empty-state">
                <i>📊</i>
                <h3>No Logs Found</h3>
                <p style="font-size: 13px; margin: 0;">Start the camera on the Live Dashboard to generate session logs.</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "⚙️ Settings":
    st.markdown("""
        <div class="header-container">
            <div class="header-title">
                <span>Settings & Info</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown("### Model Information")
    st.write("- **Engine**: Convolutional Neural Network (TensorFlow/Keras)")
    st.write("- **Face Detection**: OpenCV Haar Cascade")
    st.write("- **Input**: 48x48 Grayscale")
    st.write("- **Outputs**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
    st.markdown('</div>', unsafe_allow_html=True)
