import streamlit as st
import pandas as pd
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

# ----------------------------- MODEL DOWNLOAD & LOAD --------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... please wait ⏳"):
        url = "https://drive.google.com/uc?id=1dry3hG2bSmKQgFRK4K4REkf07JE1hcCv"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------- NUTRITION DATA (ROBUST) ------------------------
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv('nutrition_data.csv')

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns to match app format
    df = df.rename(columns={
        'food_class': 'food_name',
        'calories_per_100g': 'calories',
        'protein_g_per_100g': 'protein',
        'carbs_g_per_100g': 'carbs',
        'fat_g_per_100g': 'fat'
    })

    # Validate required columns
    required_cols = ['food_name', 'calories', 'protein', 'carbs', 'fat']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    return df.set_index('food_name').to_dict(orient='index')

NUTRIENT_DB = load_nutrition_data()

DEFAULT_NUTRIENTS = {
    'calories': 200,
    'protein': 5,
    'carbs': 20,
    'fat': 10
}

# ----------------------------- SESSION STATE ----------------------------------
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

# ----------------------------- ANALYZER PAGE ----------------------------------
def analyzer():
    st.title("🍽️ Food Analyzer")

    uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Identify Food", type="primary"):
            with st.spinner("AI is analyzing... 🤖"):
                results = model(image, imgsz=200)

                if results[0].probs is None:
                    st.error("❌ No food detected. Try another image.")
                    return

                probs = results[0].probs
                top1_idx = probs.top1
                confidence = probs.top1conf.item()
                class_name = results[0].names[top1_idx]

                st.session_state.detected_food = class_name
                st.session_state.confidence = confidence

                st.success(f"### 🎯 Identified: **{class_name.replace('_', ' ').title()}**")
                st.info(f"Confidence: {confidence:.1%}")

    # ---------------- NUTRITION CALCULATOR (ROBUST MATCHING) ----------------
    if "detected_food" in st.session_state:
        st.markdown("### ⚖️ Nutrition Calculator")

        food_key = st.session_state.detected_food  # keep original case

        # Exact match first
        if food_key in NUTRIENT_DB:
            nutrients = NUTRIENT_DB[food_key]
        else:
            # Fallback substring matching
            matched_key = None
            for db_key in NUTRIENT_DB:
                if db_key in food_key or food_key in db_key:
                    matched_key = db_key
                    break
            if matched_key:
                nutrients = NUTRIENT_DB[matched_key]
                st.info(f"Using nutrition data for '{matched_key}'")
            else:
                nutrients = DEFAULT_NUTRIENTS
                st.warning(f"Nutrition data for '{food_key}' not found, using default values.")

        weight = st.number_input("Weight (grams)", min_value=1, max_value=1000, value=150, step=10)

        if st.button("Calculate Nutrition", type="primary"):
            factor = weight / 100

            calories = round(nutrients['calories'] * factor, 1)
            protein = round(nutrients['protein'] * factor, 1)
            carbs = round(nutrients['carbs'] * factor, 1)
            fat = round(nutrients['fat'] * factor, 1)

            # Save to history
            st.session_state.history.append({
                'Food': food_key.replace('_', ' ').title(),
                'Weight (g)': weight,
                'Calories (kcal)': calories,
                'Protein (g)': protein,
                'Carbs (g)': carbs,
                'Fat (g)': fat,
                'Time': datetime.now().strftime("%H:%M:%S")
            })

            st.success("### 📊 Results")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("🔥 Calories", f"{calories} kcal")
            with col_b:
                st.metric("💪 Protein", f"{protein} g")
            with col_c:
                st.metric("🍚 Carbs", f"{carbs} g")
            with col_d:
                st.metric("🥑 Fat", f"{fat} g")

            # Optional macro bar chart
            st.bar_chart(pd.DataFrame({
                "value": [protein, carbs, fat]
            }, index=["Protein", "Carbs", "Fat"]))

# ----------------------------- INSIGHTS PAGE ----------------------------------
def insights():
    st.title("📊 Insights")

    if not st.session_state.history:
        st.info("No data yet. Go to the Analyzer and calculate some nutrition!")
        return

    df = pd.DataFrame(st.session_state.history)

    # Show full history table
    st.subheader("📜 Today's Food Log")
    st.dataframe(df, use_container_width=True)

    # Total calories summary
    total_cals = df['Calories (kcal)'].sum()
    st.info(f"📊 Total calories today: {total_cals:.0f} kcal")

    # Bar chart of total calories per food
    st.subheader("Total Calories per Food")
    total_cal_per_food = df.groupby("Food")["Calories (kcal)"].sum().sort_values(ascending=False)
    st.bar_chart(total_cal_per_food)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Log (CSV)", csv, "food_log.csv")

# ----------------------------- ROUTER -----------------------------------------
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Analyzer":
    analyzer()
else:
    insights()
