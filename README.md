# 🚀 StrataStepAI: AI-Driven Career Coach & Placement Predictor

![StrataStepAI](https://img.shields.io/badge/Status-Prototype-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B) ![Gemini API](https://img.shields.io/badge/AI-Google_Gemini_2.0-orange)

**StrataStepAI** is an intelligent web application designed to help students and job seekers evaluate their placement readiness and receive personalized, actionable career roadmaps. It combines a Machine Learning prediction model with the power of Google's Gemini Large Language Model.

## ✨ Features

* **📊 Placement Probability Checker:** Evaluates academic scores, internship counts, projects, and soft skills using a trained Random Forest model to predict the percentage probability of getting placed.
* **📄 Smart Resume Parsing:** Automatically extracts text, skills, and project details from uploaded PDF resumes.
* **🧭 AI Career Recommendation Engine:** Compares the user's resume against their desired job role using the Gemini 2.0 Flash API to generate:
    * A roadmap for missing domain-specific skills.
    * 3 tailored project ideas to strengthen the portfolio.
    * Specific certification recommendations.
* **🔄 Dynamic Session Reset:** One-click clear function to easily test multiple profiles.

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas
* **Generative AI:** Google GenAI SDK (`gemini-2.0-flash`)
* **File Processing:** PyPDF2
* **Environment Management:** python-dotenv

## 📁 Project Structure

```text
StrataStepAI/
│
├── app.py                # Main Streamlit application UI and routing
├── ml_model.py           # Script to train the ML model and generate model.pkl
├── recommendation.py     # Resume parsing and Gemini API integration logic
├── placementdata.csv     # Historical dataset used for training the ML model
├── requirements.txt      # Python dependencies (Libraries)
└── .env                  # Environment variables (For API storing)
