# 🏟️ Smart Stadium AI

An AI-powered system to improve the physical event experience in large stadiums by analyzing crowd density in real-time using computer vision.

---

## 🚀 Features

- 👥 Real-time crowd detection using YOLOv8  
- 🌡️ Heatmap visualization of crowd density  
- 📍 Zone-based analysis (Left, Center, Right)  
- 🎯 Smart gate recommendation system  
- 📊 Live dashboard with crowd analytics  

---

## 🧠 How It Works

1. Upload a stadium video  
2. The system detects people using YOLO  
3. Crowd density is calculated  
4. Heatmap is generated  
5. Least crowded zone is identified  
6. Best gate is recommended  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- OpenCV  
- YOLOv8 (Ultralytics)  
- NumPy, Pandas  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
