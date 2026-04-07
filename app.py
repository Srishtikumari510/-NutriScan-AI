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
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- CUSTOM CSS (Professional & Trendy) -------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f5 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: none;
        box-shadow: 4px 0 20px rgba(0,0,0,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 28px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
        border: 1px solid rgba(203,213,225,0.4);
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 30px -12px rgba(0,0,0,0.1);
        border-color: #cbd5e1;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 24px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(95deg, #0f2b3d 0%, #1b4a6e 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(95deg, #1b4a6e 0%, #0f2b3d 100%);
        box-shadow: 0 10px 20px -5px rgba(0,0,0,0.15);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h1 {
        background: linear-gradient(120deg, #0f2b3d, #2c7da0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Upload box */
    [data-testid="stFileUploader"] {
        background: #fef9e8;
        border-radius: 32px;
        border: 2px dashed #cbd5e1;
        padding: 1rem;
    }
    
    hr {
        margin: 2rem 0;
        background: #e2e8f0;
    }
    
    /* Success / info boxes */
    .stAlert {
        border-radius: 20px;
        border-left-width: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- MODEL & DATA LOADING --------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📡 Downloading AI model (first run only)..."):
            url = "https://drive.google.com/uc?id=1dry3hG2bSmKQgFRK4K4REkf07JE1hcCv"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = download_and_load_model()

@st.cache_data
def load_nutrition_data():
    try:
        df = pd.read_csv('nutrition_data.csv')
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={
            'food_class': 'food_name',
            'calories_per_100g': 'calories',
            'protein_g_per_100g': 'protein',
            'carbs_g_per_100g': 'carbs',
            'fat_g_per_100g': 'fat'
        })
        required = ['food_name', 'calories', 'protein', 'carbs', 'fat']
        for col in required:
            if col not in df.columns:
                st.error(f"Missing column: {col} in nutrition CSV")
                st.stop()
        return df.set_index('food_name').to_dict(orient='index')
    except Exception as e:
        st.error(f"Error loading nutrition data: {e}")
        st.stop()

NUTRIENT_DB = load_nutrition_data()
DEFAULT_NUTRIENTS = {'calories': 200, 'protein': 5, 'carbs': 20, 'fat': 10}

# ----------------------------- SESSION STATE INIT ----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'daily_goal_kcal' not in st.session_state:
    st.session_state.daily_goal_kcal = 2000
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'detected_food' not in st.session_state:
    st.session_state.detected_food = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# ----------------------------- SIDEBAR NAVIGATION ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/706/706830.png", width=60)
    st.markdown("## 🧠 NutriVision")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🍽️ Food Analyzer", "📊 Insights & History"],
        index=["🏠 Home", "🍽️ Food Analyzer", "📊 Insights & History"].index(st.session_state.page),
        label_visibility="collapsed"
    )
    st.session_state.page = page
    st.markdown("---")
    st.caption("v2.0 · AI Food Recognition")

# ----------------------------- PAGE: HOME (Landing) --------------------------
def show_home():
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good morning ☀️"
    elif current_hour < 18:
        greeting = "Good afternoon 🌤️"
    else:
        greeting = "Good evening 🌙"
    
    st.markdown(f"# {greeting}, food explorer! 👋")
    st.markdown("## Welcome to **NutriVision AI** – Your intelligent nutrition companion")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📸 What we do</h3>
        <p>Snap a photo of any meal, and our deep learning model instantly recognizes the food.  
        We then compute <strong>calories, protein, carbs, and fat</strong> based on precise nutritional data.</p>
        <p>✓ Track your daily intake  ✓ Get macro insights  ✓ Achieve your health goals</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center">
        <h3>✨ Features</h3>
        <p>🍎 100+ food classes<br>⚡ Real‑time AI<br>📈 Interactive charts<br>📅 Food log history<br>🎯 Daily calorie goal</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 How it works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Upload** – take or select a food image")
    with col_b:
        st.markdown("**2. Detect** – AI identifies the dish")
    with col_c:
        st.markdown("**3. Analyse** – get full nutrition & track it")
    
    st.markdown("---")
    st.info("💡 **Tip:** Use clear, well-lit photos of a single food item for best accuracy.")
    
    if st.button("👉 Start analyzing now", type="primary", use_container_width=False):
        st.session_state.page = "🍽️ Food Analyzer"
        st.rerun()

# ----------------------------- PAGE: FOOD ANALYZER ---------------------------
def show_analyzer():
    st.markdown("## 🍽️ AI Food Analyzer")
    st.markdown("Upload a photo, and our AI will detect the food and calculate nutrition.")
    
    left, right = st.columns([1, 1], gap="large")
    
    with left:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your meal", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file and st.button("🔍 Identify Food", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing..."):
                    results = model(image, imgsz=224)
                    if results[0].probs is None:
                        st.error("No food detected. Try a different image.")
                    else:
                        probs = results[0].probs
                        top1_idx = probs.top1
                        confidence = probs.top1conf.item()
                        class_name = results[0].names[top1_idx]
                        st.session_state.detected_food = class_name
                        st.session_state.confidence = confidence
                        st.success(f"### 🎯 Identified: **{class_name.replace('_', ' ').title()}**")
                        st.info(f"Confidence: {confidence:.1%}")
                        st.rerun()
    
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚖️ Nutrition Calculator")
        
        if st.session_state.detected_food:
            food_key = st.session_state.detected_food
            if food_key in NUTRIENT_DB:
                nutrients = NUTRIENT_DB[food_key]
            else:
                matched = None
                for db_key in NUTRIENT_DB:
                    if db_key in food_key or food_key in db_key:
                        matched = db_key
                        break
                if matched:
                    nutrients = NUTRIENT_DB[matched]
                    st.info(f"Using data for '{matched}'")
                else:
                    nutrients = DEFAULT_NUTRIENTS
                    st.warning("Using estimated nutrition values.")
            
            weight = st.number_input("Portion weight (grams)", min_value=1, max_value=1000, value=150, step=10)
            
            if st.button("📊 Calculate Nutrition", type="primary", use_container_width=True):
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
                    'Date': datetime.now().strftime("%Y-%m-%d"),
                    'Time': datetime.now().strftime("%H:%M:%S")
                })
                
                st.success("### 📊 Nutrition Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("🔥 Calories", f"{calories} kcal")
                with col2: st.metric("💪 Protein", f"{protein} g")
                with col3: st.metric("🍚 Carbs", f"{carbs} g")
                with col4: st.metric("🥑 Fat", f"{fat} g")
                
                # Native Streamlit bar chart for macros
                macro_df = pd.DataFrame({
                    'Nutrient': ['Protein', 'Carbs', 'Fat'],
                    'Grams': [protein, carbs, fat]
                })
                st.bar_chart(macro_df.set_index('Nutrient'), use_container_width=True)
                
                # Daily goal progress
                today_cals = sum(entry['Calories (kcal)'] for entry in st.session_state.history 
                                 if entry['Date'] == datetime.now().strftime("%Y-%m-%d"))
                remaining = max(0, st.session_state.daily_goal_kcal - today_cals)
                st.progress(min(1.0, today_cals / st.session_state.daily_goal_kcal))
                st.caption(f"Today's total: {today_cals:.0f} / {st.session_state.daily_goal_kcal} kcal  |  Remaining: {remaining:.0f} kcal")
                
        else:
            st.info("👈 Upload an image and click 'Identify Food' to get started.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.detected_food:
        st.markdown("---")
        st.subheader("📋 Nutritional profile (per 100g)")
        food_key = st.session_state.detected_food
        if food_key in NUTRIENT_DB:
            base = NUTRIENT_DB[food_key]
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("🔥 Base calories", f"{base['calories']} kcal")
            with col_b: st.metric("💪 Protein", f"{base['protein']} g")
            with col_c: st.metric("🍚 Carbs / 🥑 Fat", f"{base['carbs']} g / {base['fat']} g")
        else:
            st.caption("Detailed per‑100g data not available for this item.")

# ----------------------------- PAGE: INSIGHTS & HISTORY ----------------------
def show_insights():
    st.markdown("## 📊 Insights & Food Log")
    
    col_goal, col_clear = st.columns([2, 1])
    with col_goal:
        new_goal = st.number_input("🎯 Set daily calorie goal", min_value=500, max_value=5000, 
                                   value=st.session_state.daily_goal_kcal, step=50)
        if new_goal != st.session_state.daily_goal_kcal:
            st.session_state.daily_goal_kcal = new_goal
            st.success(f"Goal updated to {new_goal} kcal")
    with col_clear:
        if st.button("🗑️ Clear all history", type="secondary"):
            st.session_state.history = []
            st.rerun()
    
    if not st.session_state.history:
        st.info("No food logs yet. Go to the **Food Analyzer** page and add your first meal.")
        return
    
    df_history = pd.DataFrame(st.session_state.history)
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_today = df_history[df_history['Date'] == today_str]
    
    st.markdown("### Today's intake")
    if not df_today.empty:
        total_cals = df_today['Calories (kcal)'].sum()
        total_protein = df_today['Protein (g)'].sum()
        total_carbs = df_today['Carbs (g)'].sum()
        total_fat = df_today['Fat (g)'].sum()
        
        met1, met2, met3, met4 = st.columns(4)
        with met1: st.metric("🔥 Total calories", f"{total_cals:.0f} kcal", delta=f"{total_cals - st.session_state.daily_goal_kcal:.0f} to goal")
        with met2: st.metric("💪 Total protein", f"{total_protein:.1f} g")
        with met3: st.metric("🍚 Total carbs", f"{total_carbs:.1f} g")
        with met4: st.metric("🥑 Total fat", f"{total_fat:.1f} g")
        
        # Macro breakdown as horizontal bar chart (native)
        macro_today = pd.DataFrame({
            'Macro': ['Protein', 'Carbs', 'Fat'],
            'Grams': [total_protein, total_carbs, total_fat]
        }).set_index('Macro')
        st.subheader("Macronutrient distribution (today)")
        st.bar_chart(macro_today, use_container_width=True)
        
        # Calories per meal (bar chart)
        if len(df_today) > 1:
            st.subheader("Calories per meal (today)")
            meal_chart = df_today.set_index('Time')[['Calories (kcal)']]
            st.bar_chart(meal_chart, use_container_width=True)
    else:
        st.warning("No entries recorded today. Add a meal to see insights.")
    
    st.markdown("### 📜 Complete history")
    st.dataframe(df_history.drop(columns=['Date'] if 'Date' in df_history.columns else []), 
                 use_container_width=True, height=300)
    
    csv = df_history.to_csv(index=False)
    st.download_button("📥 Download log as CSV", csv, "food_log.csv", use_container_width=True)

# ----------------------------- PAGE DISPATCHER --------------------------------
if st.session_state.page == "🏠 Home":
    show_home()
elif st.session_state.page == "🍽️ Food Analyzer":
    show_analyzer()
elif st.session_state.page == "📊 Insights & History":
    show_insights()
