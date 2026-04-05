"""
NutriScan AI - Food Nutrition Analyzer
Detects food from an image, calculates nutrition based on user-provided weight.
"""

import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------
# Load model and nutrition data (cached for performance)
# ------------------------------
@st.cache_resource
def load_model():
    """Load the YOLO classification model."""
    model_path = "best.pt"  # Make sure this file is in the same directory as app.py
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please add it to the app directory.")
        st.stop()
    return YOLO(model_path)

@st.cache_data
def load_nutrition_data():
    """Load nutrition CSV and return a dict keyed by food_class."""
    csv_path = "nutrition_data.csv"
    if not os.path.exists(csv_path):
        st.error(f"Nutrition CSV '{csv_path}' not found. Please add it to the app directory.")
        st.stop()
    df = pd.read_csv(csv_path)
    # Convert to dictionary for faster lookup
    return df.set_index("food_class").to_dict(orient="index")

# ------------------------------
# Helper functions
# ------------------------------
def predict_food(image, model):
    """Run prediction on an image and return top food name and confidence."""
    results = model(image)
    probs = results[0].probs
    top1_idx = probs.top1
    food_name = results[0].names[top1_idx]
    confidence = probs.top1conf.item()
    return food_name, confidence

def calculate_nutrition(food_name, weight_g, nutrition_dict):
    """Calculate nutrition for the given weight based on per-100g values."""
    if food_name not in nutrition_dict:
        return None
    data = nutrition_dict[food_name]
    factor = weight_g / 100.0
    return {
        "calories": data["calories_per_100g"] * factor,
        "protein": data["protein_g_per_100g"] * factor,
        "carbs": data["carbs_g_per_100g"] * factor,
        "fat": data["fat_g_per_100g"] * factor,
    }

# ------------------------------
# Main UI
# ------------------------------
st.title("🍽️ NutriScan AI")
st.markdown("### Know exactly what you eat")
st.write("Upload a photo of your meal, enter its weight, and get an instant nutritional breakdown.")

# Two-column layout for inputs
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "📸 Upload food image", 
        type=["jpg", "jpeg", "png"],
        help="Take a clear photo of your meal"
    )
with col2:
    weight = st.number_input(
        "⚖️ Meal weight (grams)", 
        min_value=0.0, 
        step=10.0,
        help="Weigh your food for accurate results"
    )

# When image and weight are provided
if uploaded_file is not None and weight > 0:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your meal", use_container_width=False, width=300)
    
    # Load model and nutrition data
    with st.spinner("🔍 Analyzing your food..."):
        model = load_model()
        nutrition_dict = load_nutrition_data()
        
        # Predict
        food_name, confidence = predict_food(image, model)
        
        # Show detection result
        st.success(f"✅ **Detected:** {food_name.replace('_', ' ').title()}")
        st.write(f"*Confidence: {confidence:.1%}*")
        
        # Calculate nutrition
        nutrition = calculate_nutrition(food_name, weight, nutrition_dict)
        
        if nutrition:
            st.subheader("📊 Nutrition Facts")
            # Display metrics in columns
            metric_cols = st.columns(4)
            metric_cols[0].metric("🔥 Calories", f"{nutrition['calories']:.0f} kcal")
            metric_cols[1].metric("💪 Protein", f"{nutrition['protein']:.1f} g")
            metric_cols[2].metric("🍚 Carbohydrates", f"{nutrition['carbs']:.1f} g")
            metric_cols[3].metric("🥑 Fat", f"{nutrition['fat']:.1f} g")
            
            # Optional: Macronutrient bar chart
            st.subheader("Macronutrient distribution")
            macro_df = pd.DataFrame({
                "Nutrient": ["Protein", "Carbs", "Fat"],
                "Grams": [nutrition["protein"], nutrition["carbs"], nutrition["fat"]]
            })
            st.bar_chart(macro_df.set_index("Nutrient"))
        else:
            st.warning(f"⚠️ Nutrition data for '{food_name}' is missing. Please add it to `nutrition_data.csv`.")
            
elif uploaded_file is not None and weight == 0:
    st.warning("Please enter the weight of your meal to calculate nutrition.")
elif uploaded_file is None and weight > 0:
    st.info("Upload an image to get started.")

# Footer
st.markdown("---")
st.caption("**Note:** Nutritional values are estimates. For medical advice, consult a professional.")
