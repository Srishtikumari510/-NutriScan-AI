import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import os
import gdown
from ultralytics import YOLO

# ----------------------------- PAGE CONFIG ------------------------------------
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- MODERN CSS -------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}

/* Hero section */
.hero {
    padding: 3rem;
    border-radius: 30px;
    background: linear-gradient(120deg, #1e3a8a, #06b6d4);
    color: white;
}

/* Buttons */
.stButton > button {
    border-radius: 30px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- MODEL ------------------------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            url = "https://drive.google.com/uc?id=1dry3hG2bSmKQgFRK4K4REkf07JE1hcCv"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------- DATA -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("nutrition_data.csv")
    df.columns = df.columns.str.lower()
    return df.set_index("food_class").to_dict("index")

DB = load_data()

# ----------------------------- SESSION ----------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "history" not in st.session_state:
    st.session_state.history = []
if "name" not in st.session_state:
    st.session_state.name = ""

# ----------------------------- SIDEBAR ----------------------------------------
with st.sidebar:
    st.title("🥗 NutriVision")
    st.session_state.name = st.text_input("Your Name")

    page = st.radio("Menu", ["Home", "Analyzer", "Insights"])
    st.session_state.page = page

# ----------------------------- HOME PAGE --------------------------------------
def home():
    name = st.session_state.name or "there"

    st.markdown(f"""
    <div class="hero">
        <h1>👋 Hello {name}!</h1>
        <h2>Your AI-powered Nutrition Assistant</h2>
        <p>Track calories, analyze meals, and stay healthy with AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🚀 What We Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">📸 Upload food image</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">🤖 AI detects food</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">📊 Get nutrition insights</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Quick Health Tool")

    weight = st.number_input("Weight (kg)", 30, 150, 60)
    height = st.number_input("Height (cm)", 100, 220, 165)

    if st.button("Calculate BMI"):
        bmi = weight / ((height/100)**2)
        st.success(f"Your BMI: {bmi:.1f}")

    if st.button("Start Analyzing 🍽️"):
        st.session_state.page = "Analyzer"
        st.rerun()

# ----------------------------- ANALYZER ---------------------------------------
def analyzer():
    st.title("🍽️ Food Analyzer")

    file = st.file_uploader("Upload food image")

    if file:
        img = Image.open(file)
        st.image(img)

        if st.button("Analyze"):
            res = model(img)

            if res[0].probs:
                idx = res[0].probs.top1
                food = res[0].names[idx]
                conf = res[0].probs.top1conf.item()

                st.success(f"{food} ({conf:.1%})")

                nutrients = DB.get(food, {"calories":200,"protein":5,"carbs":20,"fat":10})

                weight = st.slider("Weight (g)", 50, 500, 150)

                if st.button("Calculate"):
                    factor = weight/100

                    cal = nutrients["calories"]*factor
                    pro = nutrients["protein"]*factor
                    carb = nutrients["carbs"]*factor
                    fat = nutrients["fat"]*factor

                    st.metric("Calories", cal)
                    st.metric("Protein", pro)
                    st.metric("Carbs", carb)
                    st.metric("Fat", fat)

                    st.bar_chart(pd.DataFrame({
                        "value":[pro,carb,fat]
                    }, index=["Protein","Carbs","Fat"]))

                    st.session_state.history.append({
                        "food":food,
                        "cal":cal
                    })

# ----------------------------- INSIGHTS ---------------------------------------
def insights():
    st.title("📊 Insights")

    if not st.session_state.history:
        st.info("No data yet")
        return

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df)

    st.bar_chart(df.set_index("food"))

# ----------------------------- ROUTER -----------------------------------------
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Analyzer":
    analyzer()
else:
    insights()
