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

# ----------------------------- DATA (FIXED) -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("nutrition_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        'food_class': 'food_class',
        'calories_per_100g': 'calories',
        'protein_g_per_100g': 'protein',
        'carbs_g_per_100g': 'carbs',
        'fat_g_per_100g': 'fat'
    })
    return df.set_index("food_class").to_dict("index")

DB = load_data()   # ✅ IMPORTANT: actually call the function

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

    uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Identify Food"):
            with st.spinner("AI is analyzing... 🤖"):
                results = model(image, imgsz=224)

                if results[0].probs is None:
                    st.error("❌ No food detected. Try another image.")
                    return

                probs = results[0].probs
                top1_idx = probs.top1
                confidence = probs.top1conf.item()
                class_name = results[0].names[top1_idx]

                st.session_state.detected_food = class_name
                st.session_state.confidence = confidence

                st.success(f"🎯 {class_name.replace('_',' ').title()} ({confidence:.1%})")

    # ---------------- NUTRITION (with robust matching) ----------------
    if "detected_food" in st.session_state:
        st.markdown("### ⚖️ Nutrition Calculator")

        food_key = st.session_state.detected_food.lower().strip()

        # ✅ Improved matching (from first code block)
        if food_key in DB:
            nutrients = DB[food_key]
        else:
            match = None
            for k in DB:
                if k in food_key or food_key in k:
                    match = k
                    break
            if match:
                nutrients = DB[match]
                st.info(f"Using data for {match}")
            else:
                nutrients = {
                    "calories": 200,
                    "protein": 5,
                    "carbs": 20,
                    "fat": 10
                }
                st.warning("Using default nutrition values")

        weight = st.slider("Weight (grams)", 50, 500, 150)

        if st.button("Calculate Nutrition"):
            factor = weight / 100

            # ✅ Safe access with .get()
            cal = nutrients.get("calories", 200) * factor
            pro = nutrients.get("protein", 5) * factor
            carb = nutrients.get("carbs", 20) * factor
            fat = nutrients.get("fat", 10) * factor

            st.metric("🔥 Calories", f"{cal:.1f}")
            st.metric("💪 Protein", f"{pro:.1f}")
            st.metric("🍚 Carbs", f"{carb:.1f}")
            st.metric("🥑 Fat", f"{fat:.1f}")

            st.bar_chart(pd.DataFrame({
                "value": [pro, carb, fat]
            }, index=["Protein", "Carbs", "Fat"]))

            # ✅ Add timestamp to history
            st.session_state.history.append({
                "food": food_key,
                "calories": cal,
                "time": datetime.now().strftime("%H:%M:%S")
            })

# ----------------------------- INSIGHTS ---------------------------------------
def insights():
    st.title("📊 Insights")

    if not st.session_state.history:
        st.info("No data yet. Go to the Analyzer and calculate some nutrition!")
        return

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # Show total calories per food item
    st.subheader("Total Calories per Food")
    total_cal = df.groupby("food")["calories"].sum().sort_values(ascending=False)
    st.bar_chart(total_cal)

# ----------------------------- ROUTER -----------------------------------------
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Analyzer":
    analyzer()
else:
    insights()
