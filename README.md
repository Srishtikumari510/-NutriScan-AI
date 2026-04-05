# 🍽️ NutriScan AI - Food Nutrition Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📌 Overview
NutriScan AI is an intelligent web application that identifies food from an image and calculates its nutritional content (calories, protein, carbs, fat) based on the user-provided weight. It uses a YOLOv8 classification model trained on the Food-101 dataset (101 food categories).

## ✨ Features
- 🔍 **Food Recognition** – Identifies 101 different foods from a photo
- ⚖️ **Weight‑Based Nutrition** – User enters weight (grams), app computes per‑portion values
- 📊 **Instant Results** – Displays calories, protein, carbs, fat with a macronutrient bar chart
- 🎯 **Confidence Score** – Shows how sure the model is about its prediction
- 🚀 **Easy to Use** – Simple upload and weight input

## 🛠️ How It Works
1. User uploads a clear photo of a meal.
2. User enters the approximate weight of the meal (in grams).
3. The YOLOv8 model predicts the food category.
4. The app looks up nutrition per 100g from a CSV database.
5. Results are scaled to the entered weight and displayed.

## 📁 Project Structure
├── app.py # Streamlit application
├── best.pt # Trained YOLOv8 classification model
├── nutrition_data.csv # Nutrition database (101 foods)
├── requirements.txt # Python dependencies
└── README.md # This file


## 🚀 Deployment
This app is deployed on **Streamlit Cloud**. To deploy your own instance:

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`
4. Or deploy to Streamlit Cloud by connecting your GitHub repo.

## 📊 Dataset & Model
- **Dataset:** [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) – 101,000 images of 101 food categories.
- **Model:** YOLOv8n-cls (classification) trained for 5 epochs (63.6% top‑1 accuracy, 86.3% top‑5). For better accuracy, retrain with more epochs.
- **Nutrition Data:** Gathered from public sources (USDA, nutrition databases). Approximate values – refine as needed.

## 🧪 Example
**Input:** Photo of pizza + weight 200g  
**Output:**  
- Detected: Pizza (85% confidence)  
- Calories: 532 kcal  
- Protein: 22g  
- Carbs: 66g  
- Fat: 20g  

## 📝 Future Improvements
- Multi‑food detection (recognize several items on a plate)
- Barcode scanning for packaged foods
- Meal history and daily tracking
- Voice input for hands‑free use
- Export reports as PDF

## 🤝 Contributing
Feel free to open issues or pull requests to improve the nutrition database or model accuracy.

## 📄 License
MIT License – free for personal and educational use.

## 🙏 Acknowledgements
- Ultralytics for YOLOv8
- Food-101 dataset creators
- Streamlit for the awesome framework

---
**Made with ❤️ for healthier eating**
