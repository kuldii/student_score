import os
import gdown
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------
# Download model files from Google Drive
# ---------------------------------------------
MODELS_FILE_ID = "1_7CSZG3g77yzL0AcK-sHn-EMNDYL9ex7"
PREPROCESSORS_FILE_ID = "10yRAj2wHhpAEiwLYbRl7pO1ufGTK4iTn"

os.makedirs("models", exist_ok=True)

model_path = "models/student_models.pkl"
preprocessor_path = "models/preprocessors.pkl"

if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={MODELS_FILE_ID}", model_path, quiet=False)

if not os.path.exists(preprocessor_path):
    gdown.download(f"https://drive.google.com/uc?id={PREPROCESSORS_FILE_ID}", preprocessor_path, quiet=False)

# ---------------------------------------------
# Load models and preprocessors
# ---------------------------------------------
model_dict = joblib.load(model_path)
preprocessors = joblib.load(preprocessor_path)

ohe = preprocessors['encoder']
scaler = preprocessors['scaler']
options = preprocessors['options']

# ---------------------------------------------
# Preprocessing function
# ---------------------------------------------
def preprocess_input(gender, race, parental, lunch, prep):
    input_df = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [parental],
        'lunch': [lunch],
        'test preparation course': [prep]
    })
    input_encoded = ohe.transform(input_df)
    input_scaled = scaler.transform(input_encoded)
    return input_scaled

# ---------------------------------------------
# Prediction function
# ---------------------------------------------
def predict_scores(model_name, gender, race, parental, lunch, prep):
    model = model_dict[model_name]
    X_input = preprocess_input(gender, race, parental, lunch, prep)
    pred = model.predict(X_input)[0]
    return round(pred[0], 2), round(pred[1], 2), round(pred[2], 2)

# ---------------------------------------------
# Streamlit App UI
# ---------------------------------------------
st.set_page_config(page_title="Student Exam Score Prediction", page_icon="🎓", layout="centered")

st.title("🎓 Student Exam Score Prediction")
st.markdown("""
**Powered by Scikit-learn & Streamlit — by Kuldii Project**

This application predicts a student's exam performance in **Math**, **Reading**, and **Writing** based on demographic and academic preparation data.
""")

with st.expander("🧾 How it Works", expanded=True):
    st.markdown("""
    ### 📋 Input Features
    - **Gender** — Select the student's gender.
    - **Race/Ethnicity** — Choose from demographic categories.
    - **Parental Level of Education** — Parent's highest education level.
    - **Lunch Type** — Whether the student receives standard or free/reduced lunch.
    - **Test Preparation Course** — Completed or not.

    ### 🤖 Model Selection
    Choose one of the trained regression models:
    - **Linear Regression**
    - **Random Forest**
    - **Gradient Boosting**

    Once the inputs are filled, click **📈 Predict Scores** to view the results.
    """)

st.divider()

# ---------------------------------------------
# Form input
# ---------------------------------------------
with st.form("prediction_form"):
    st.subheader("📋 Student Information")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("👤 Gender", options['gender'])
        race = st.selectbox("🌍 Race/Ethnicity", options['race/ethnicity'])
    with col2:
        parental = st.selectbox("🎓 Parental Education", options['parental level of education'])
        lunch = st.selectbox("🍽️ Lunch Type", options['lunch'])

    prep = st.selectbox("📘 Test Preparation Course", options['test preparation course'])
    
    st.subheader("🤖 Model Selection")
    model_name = st.selectbox("🧠 Select Model", list(model_dict.keys()))
    
    submitted = st.form_submit_button("📈 Predict Scores")

# ---------------------------------------------
# Output prediction
# ---------------------------------------------
if submitted:
    math, reading, writing = predict_scores(model_name, gender, race, parental, lunch, prep)

    st.success("### ✅ Predicted Exam Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Math Score", f"{math}")
    col2.metric("📖 Reading Score", f"{reading}")
    col3.metric("✍️ Writing Score", f"{writing}")
