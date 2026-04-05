import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Load nutrition data from CSV
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv('nutrition_data.csv')
    # Convert to dictionary for faster lookup
    return df.set_index('food_name').to_dict(orient='index')

NUTRIENT_DB = load_nutrition_data()
DEFAULT_NUTRIENTS = {'calories': 200, 'protein': 5, 'carbs': 20, 'fat': 10, 'fiber': 0}

st.set_page_config(page_title="AI Food Nutrition Analyzer", layout="wide")
st.title("🍽️ AI Food Nutrition Analyzer")
st.markdown("Upload a photo – AI will identify the food and calculate nutrition.")

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("🔍 Identify Food", type="primary"):
            with st.spinner("AI is analyzing..."):
                results = model(image, imgsz=224)
                probs = results[0].probs
                top1_idx = probs.top1
                confidence = probs.top1conf.item()
                class_name = results[0].names[top1_idx]

                st.success(f"### 🎯 Identified: **{class_name.replace('_', ' ').title()}**")
                st.info(f"Confidence: {confidence:.1%}")

                st.session_state.detected_food = class_name
                st.session_state.confidence = confidence

with col2:
    st.subheader("⚖️ Nutrition Calculator")
    if 'detected_food' in st.session_state:
        food_key = st.session_state.detected_food
        # Try exact match first, then substring match
        if food_key in NUTRIENT_DB:
            nutrients = NUTRIENT_DB[food_key]
        else:
            # Fallback: find any key that contains food_key or vice versa
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
            with col_a: st.metric("🔥 Calories", f"{calories} kcal")
            with col_b: st.metric("💪 Protein", f"{protein} g")
            with col_c: st.metric("🍚 Carbs", f"{carbs} g")
            with col_d: st.metric("🥑 Fat", f"{fat} g")

if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Today's Food Log")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    total_cals = df['Calories (kcal)'].sum()
    st.info(f"📊 Total calories today: {total_cals:.0f} kcal")
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Log", csv, "food_log.csv")
