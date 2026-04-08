 🥗 NutriVision AI

AI-Powered Food Recognition & Nutrition Analyzer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nutrivision-ai.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Srishtikumari510/NutriVision-AI)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🚀 Live App: [nutrivision-ai.streamlit.app](https://lntcnd8plgtvssafx7dvrq.streamlit.app/)  
📂 GitHub Repo: [github.com/Srishtikumari510/NutriVision-AI](https://github.com/Srishtikumari510/NutriVision-AI)

---

 📌 Overview

NutriVision AI is an intelligent web application that uses deep learning to identify food items from images and instantly provide nutritional insights like calories, protein, carbohydrates, and fats.

Built using Streamlit and a custom-trained YOLO model, this project helps users track their diet and make healthier decisions in real-time – no manual food logging required.

---

 ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📸 Food Image Upload | Upload JPG, JPEG, or PNG images |
| 🤖 AI Food Detection | YOLOv8-based model for accurate classification |
| ⚖️ Dynamic Nutrition Calculator | Adjust portion size (grams) – nutrients update instantly |
| 📊 Nutritional Insights | Calories, Protein, Carbs, Fat breakdown + progress bars |
| 📈 Daily Food Log | Saves all analyzed foods in your browser session |
| 📥 Export Data | Download your food log as CSV for tracking |
| 🧮 BMI Calculator | Quick health metric on the homepage |
| 🎨 Modern UI | Glassmorphism design, responsive layout, dark/light friendly |

---

 🧠 Tech Stack

| Category         | Technology                       |
| ---------------- | -------------------------------- |
| Frontend         | Streamlit                        |
| Backend          | Python 3.9+                      |
| AI Model         | YOLOv8 (Ultralytics)             |
| Data Handling    | Pandas                           |
| Image Processing | Pillow (PIL)                     |
| Model Hosting    | Google Drive (via `gdown`)       |
| Deployment       | Streamlit Cloud                  |

---

 📂 Project Structure

NutriVision-AI/
│
├── app.py  Main Streamlit application
├── best.pt  YOLO trained model (auto-downloaded on first run)
├── nutrition_data.csv  Food nutrition dataset (calories, protein, etc.)
├── requirements.txt  Python dependencies
└── README.md  Project documentation

---

 ⚙️ Installation & Setup

 1️⃣ Clone the Repository

```bash
git clone https://github.com/Srishtikumari510/NutriVision-AI.git
cd NutriVision-AI
```

 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       Mac/Linux
venv\Scripts\activate          Windows
```

 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

 4️⃣ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

---

 ☁️ Deployment (Streamlit Cloud)

Deploy your own instance in minutes:

1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New app** → connect your repo.
4. Set main file path to `app.py`.
5. Click **Deploy**.

💡 The model will be downloaded automatically from Google Drive when the app runs for the first time on Streamlit's servers.

---

 🧠 Model Information

- **Model:** YOLOv8n (nano) fine-tuned on a custom food dataset
- **Training Platform:** Google Colab with GPU (T4)
- **Dataset:** 15 food classes (pizza, burger, salad, apple, etc.)
- **Download Method:** Google Drive ID embedded in `app.py` – downloaded using `gdown` if `best.pt` not found.

```python
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1ABC123xyz..."  
    gdown.download(url, MODEL_PATH, quiet=False)
```

---

 📊 How It Works

1. User uploads an image of a single food item.
2. Image is preprocessed and passed to YOLO model.
3. Model returns predicted class name + confidence score.
4. App looks up nutrition data from `nutrition_data.csv`.
5. User adjusts portion size (in grams) using a slider.
6. Nutritional values are scaled proportionally.
7. The entry is added to the session's Daily Food Log.
8. Charts and insights are updated automatically.

---

 📈 Future Enhancements

- 🍽️ Multi-food detection – recognize all items in one image
- 🎯 Personalized diet goals – set daily calorie/protein targets
- 🔥 Calorie tracking over time – line charts & streak counters
- 📱 Mobile app version – built with Flutter or React Native
- 🌐 Live nutrition API – fallback to USDA/FDC API for unknown foods
- 🧠 Improved model accuracy – train on Food-101 or larger dataset

---

 ⚠️ Limitations

- Detects only one food per image (multi-food support planned)
- Nutrition values are estimates based on standard portion sizes
- Model accuracy depends on training dataset – may misclassify similar-looking foods
- Requires internet connection for first-time model download (cached afterward)

---

 🤝 Contributing

Contributions are welcome! Whether it's a bug fix, new feature, or dataset improvement.

**Steps:**

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-idea`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-idea`
5. Open a Pull Request.

Please ensure your code follows black formatting and includes relevant docstrings.

---

 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the awesome detection framework
- [Streamlit](https://streamlit.io) for making ML apps ridiculously easy
- [Google Colab](https://colab.research.google.com) for free GPU training
- Open-source food nutrition datasets from Kaggle & USDA

---

 👩‍💻 Author

**Srishti Kumari**  
[www.linkedin.com/in/srishti-kumari-335b67252](https://www.linkedin.com/in/srishti-kumari-335b67252)  
[https://github.com/Srishtikumari510](https://github.com/Srishtikumari510)

---

 💬 Connect & Support

If you found this project useful, please ⭐ star the repo and share it with others!

Have questions or suggestions? Open an issue or reach out via LinkedIn.

**Happy healthy eating! 🥗✨**
