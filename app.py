import joblib
import pandas as pd
import gradio as gr

# ---------------------------------------------
# Load Models and Preprocessors
# ---------------------------------------------
model_dict = joblib.load("models/student_models.pkl")
preprocessors = joblib.load("models/preprocessors.pkl")

ohe = preprocessors['encoder']
scaler = preprocessors['scaler']
options = preprocessors['options']

# ---------------------------------------------
# Preprocessing Function
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
# Prediction Function
# ---------------------------------------------
def predict_scores(model_name, gender, race, parental, lunch, prep):
    model = model_dict[model_name]
    X_input = preprocess_input(gender, race, parental, lunch, prep)
    pred = model.predict(X_input)[0]
    return round(pred[0], 2), round(pred[1], 2), round(pred[2], 2)

# ---------------------------------------------
# Gradio UI
# ---------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎓 Student Exam Score Prediction
    **Powered by Scikit-learn & Gradio — by Kuldii Project**

    This application predicts a student's exam performance in **Math**, **Reading**, and **Writing** based on demographic and academic preparation data.

    ### 📋 Input Features
    - **Gender** — Select the student's gender.
    - **Race/Ethnicity** — Choose from predefined demographic groups.
    - **Parental Level of Education** — Indicate the parent's highest education level.
    - **Lunch Type** — Standard or Free/Reduced lunch.
    - **Test Preparation Course** — Completed or None.

    ### 🤖 Model Selection
    Choose one of the available machine learning models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting

    Click **Predict Scores** to get the student's estimated results.
    """)

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(model_dict.keys()),
            label="🧠 Select Model",
            value=list(model_dict.keys())[0]
        )

    with gr.Row():
        gender_input = gr.Dropdown(choices=options['gender'], label="👤 Gender")
        race_input = gr.Dropdown(choices=options['race/ethnicity'], label="🌍 Race/Ethnicity")

    with gr.Row():
        parental_input = gr.Dropdown(choices=options['parental level of education'], label="🎓 Parental Education")
        lunch_input = gr.Dropdown(choices=options['lunch'], label="🍽️ Lunch Type")

    with gr.Row():
        prep_input = gr.Dropdown(choices=options['test preparation course'], label="📘 Test Preparation")

    predict_btn = gr.Button("📈 Predict Scores")

    with gr.Row():
        math_output = gr.Number(label="📊 Predicted Math Score")
        reading_output = gr.Number(label="📖 Predicted Reading Score")
        writing_output = gr.Number(label="✍️ Predicted Writing Score")

    predict_btn.click(
        fn=predict_scores,
        inputs=[model_choice, gender_input, race_input, parental_input, lunch_input, prep_input],
        outputs=[math_output, reading_output, writing_output]
    )

demo.launch()