import streamlit as st
import numpy as np
import pandas as pd
import random

st.set_page_config(layout="wide")
st.title("🏟️ Smart Stadium AI Dashboard")

video_file = st.file_uploader("📤 Upload Stadium Video", type=["mp4"])

if video_file:

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

    # Simulate 50 frames
    for i in range(50):

        # Fake AI values (simulate detection)
        total_people = random.randint(20, 120)
        left_count = random.randint(5, total_people//2)
        center_count = random.randint(5, total_people//2)
        right_count = total_people - left_count - center_count

        zones = {
            "Gate A (Left)": left_count,
            "Gate B (Center)": center_count,
            "Gate C (Right)": right_count
        }

        best_gate = min(zones, key=zones.get)

        # Fake frame (just placeholder)
        frame = np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)
        frame_window.image(frame, caption=f"People: {total_people} | Best Gate: {best_gate}")

        # Metrics
        total_placeholder.metric("👥 Total", total_people)
        left_placeholder.metric("⬅️ Left", left_count)
        center_placeholder.metric("⬆️ Center", center_count)
        right_placeholder.metric("➡️ Right", right_count)
        gate_placeholder.success(f"🎯 Use: {best_gate}")

        # Graph
        people_history.append(total_people)
        df = pd.DataFrame(people_history, columns=["People"])
        chart.line_chart(df)
