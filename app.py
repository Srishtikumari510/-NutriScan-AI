import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
import os
import gdown
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="wide"
)

# ---------------- MODEL ----------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            url = "https://drive.google.com/uc?id=1dry3hG2bSmKQgFRK4K4REkf07JE1hcCv"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- DATA ----------------
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

# ✅🔥 IMPORTANT FIX
DB = load_data()

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🥗 NutriVision")
    page = st.radio("Menu", ["Home", "Analyzer", "Insights"])
    st.session_state.page = page

# ---------------- HOME ----------------
def home():
    st.title("👋 Welcome to NutriVision AI")
    st.write("Upload food images and get nutrition instantly!")

# ---------------- ANALYZER ----------------
def analyzer():
    st.title("🍽️ Food Analyzer")

    file = st.file_uploader("Upload food image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image)

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing..."):
                results = model(image, imgsz=224)

                if results[0].probs is None:
                    st.error("No food detected")
                    return

                probs = results[0].probs
                idx = probs.top1
                food = results[0].names[idx]
                conf = probs.top1conf.item()

                st.session_state.detected_food = food.lower()
                st.success(f"{food} ({conf:.1%})")

    # -------- NUTRITION --------
    if "detected_food" in st.session_state:
        food_key = st.session_state.detected_food

        # 🔥 MATCH FIX
        if food_key in DB:
            nutrients = DB[food_key]
        else:
            nutrients = None
            for k in DB:
                if k in food_key or food_key in k:
                    nutrients = DB[k]
                    st.info(f"Using {k} data")
                    break

            if nutrients is None:
                nutrients = {"calories":200,"protein":5,"carbs":20,"fat":10}
                st.warning("Using default values")

        weight = st.slider("Weight (g)", 50, 500, 150)

        if st.button("Calculate Nutrition"):
            factor = weight / 100

            # ✅ SAFE ACCESS (NO ERROR NOW)
            cal = nutrients.get("calories", 200) * factor
            pro = nutrients.get("protein", 5) * factor
            carb = nutrients.get("carbs", 20) * factor
            fat = nutrients.get("fat", 10) * factor

            st.metric("Calories", f"{cal:.1f}")
            st.metric("Protein", f"{pro:.1f}")
            st.metric("Carbs", f"{carb:.1f}")
            st.metric("Fat", f"{fat:.1f}")

            st.session_state.history.append({
                "food": food_key,
                "calories": cal,
                "time": datetime.now().strftime("%H:%M")
            })

# ---------------- INSIGHTS ----------------
def insights():
    st.title("📊 Insights")

    if not st.session_state.history:
        st.info("No data yet")
        return

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df)
    st.bar_chart(df.set_index("food")["calories"])

# ---------------- ROUTER ----------------
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Analyzer":
    analyzer()
else:
    insights()
