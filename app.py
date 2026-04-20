import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd

# ---------------- UI CONFIG ----------------
st.set_page_config(layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>
    🏟️ Smart Stadium AI Dashboard
    </h1>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- FILE UPLOAD ----------------
video_file = st.file_uploader("📤 Upload Stadium Video", type=["mp4"])

if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # Layout
    col1, col2 = st.columns([3, 1])
    frame_window = col1.empty()

    with col2:
        st.subheader("📊 Live Metrics")
        total_placeholder = st.empty()
        left_placeholder = st.empty()
        center_placeholder = st.empty()
        right_placeholder = st.empty()
        gate_placeholder = st.empty()

    st.subheader("📈 Crowd Trend")
    chart = st.line_chart()

    people_history = []
    frame_count = 0

    # ---------------- MAIN LOOP ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        results = model(frame)

        height, width, _ = frame.shape

        total_people = 0
        left_count = 0
        center_count = 0
        right_count = 0

        heatmap = np.zeros((height, width), dtype=np.float32)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:

                    total_people += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)

                    if cx < width / 3:
                        left_count += 1
                    elif cx < 2 * width / 3:
                        center_count += 1
                    else:
                        right_count += 1

                    heatmap[y1:y2, x1:x2] += 1

        # ---------------- HEATMAP ----------------
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # ---------------- SMART GATE SYSTEM ----------------
        zones = {
            "Gate A (Left)": left_count,
            "Gate B (Center)": center_count,
            "Gate C (Right)": right_count
        }

        best_gate = min(zones, key=zones.get)

        # ---------------- DRAW TEXT ----------------
        cv2.putText(overlay, f"People: {total_people}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(overlay, f"Best Entry: {best_gate}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # ---------------- DISPLAY VIDEO ----------------
        frame_window.image(overlay, channels="BGR")

        # ---------------- UPDATE METRICS ----------------
        total_placeholder.metric("👥 Total", total_people)
        left_placeholder.metric("⬅️ Left", left_count)
        center_placeholder.metric("⬆️ Center", center_count)
        right_placeholder.metric("➡️ Right", right_count)
        gate_placeholder.success(f"🎯 Use: {best_gate}")

        # ---------------- GRAPH ----------------
        people_history.append(total_people)

        df = pd.DataFrame(people_history, columns=["People"])
        chart.line_chart(df)

    cap.release()

    st.success("✅ Analysis Complete")