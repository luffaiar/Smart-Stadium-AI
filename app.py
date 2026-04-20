import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import pandas as pd

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🏟️ Smart Stadium AI Dashboard")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- Upload Video ----------------
video_file = st.file_uploader("📤 Upload Stadium Video", type=["mp4"])

if video_file:

    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Load video using ultralytics-compatible method
    import cv2  # only used internally, safer here

    cap = cv2.VideoCapture(tfile.name)

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

    # ---------------- Processing Loop ----------------
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

        # ---------------- Smart Gate ----------------
        zones = {
            "Gate A (Left)": left_count,
            "Gate B (Center)": center_count,
            "Gate C (Right)": right_count
        }

        best_gate = min(zones, key=zones.get)

        # ---------------- Display Frame ----------------
        img = Image.fromarray(frame)
        frame_window.image(img, caption=f"People: {total_people} | Best Gate: {best_gate}")

        # ---------------- Metrics ----------------
        total_placeholder.metric("👥 Total", total_people)
        left_placeholder.metric("⬅️ Left", left_count)
        center_placeholder.metric("⬆️ Center", center_count)
        right_placeholder.metric("➡️ Right", right_count)
        gate_placeholder.success(f"🎯 Use: {best_gate}")

        # ---------------- Graph ----------------
        people_history.append(total_people)
        df = pd.DataFrame(people_history, columns=["People"])
        chart.line_chart(df)

    cap.release()

    st.success("✅ Analysis Complete")
