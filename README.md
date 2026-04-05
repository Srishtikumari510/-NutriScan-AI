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
