# Student Exam Score Prediction App

A professional, production-ready machine learning app for predicting student exam scores (math, reading, writing) using the Students Performance dataset. Built with robust preprocessing, multiple regression models, and a modern Streamlit UI. Fully containerized for easy deployment.

---

## 🚀 Features

- **Robust Preprocessing**: Categorical encoding, feature scaling, and outlier detection
- **Multiple Regression Models**: Linear Regression, Random Forest, Gradient Boosting (with hyperparameter tuning)
- **Rich Visualizations & EDA**: Boxplots, histograms, pairplots, correlation heatmap, feature importance
- **Interactive Streamlit UI**: User-friendly dropdowns, model selection, and instant predictions
- **Production-Ready**: Dockerized, reproducible environment, and easy deployment

---

## 🏗️ Project Structure

```
├── app.py                      # Gradio app for prediction (legacy)
├── streamlit_app.py            # Streamlit app for prediction (production-ready)
├── student_exam_score_prediction.ipynb  # Full EDA, modeling, and training notebook
├── models/
│   ├── student_models.pkl      # Trained regression models (joblib)
│   └── preprocessors.pkl       # Preprocessing tools and dropdown options
├── environment.yml             # Conda environment for reproducibility
├── Dockerfile                  # Containerization for deployment
└── README.md                   # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data)
- **Preprocessing**:
  - Outlier detection for scores
  - One-hot encoding for categorical features
  - Standardization of numeric features

---

## 🧠 Models

- **Linear Regression**
- **Random Forest Regressor** (with GridSearchCV)
- **Gradient Boosting Regressor** (with GridSearchCV)

All models are trained and saved for instant prediction in the app.

---

## 🖥️ Streamlit App

- **Dropdowns** for all features (gender, race/ethnicity, parental education, lunch, test preparation)
- **Model selection** dropdown
- **Prediction output**: Estimated math, reading, and writing scores
- **Production config**: Ready for Docker deployment

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kuldii/student_score.git
cd student_score
```

### 2. Install Dependencies
```bash
conda env create -f environment.yml
conda activate student-score-env
```

### 3. (Optional) Train Models
- All models and preprocessors are pre-trained and saved in `models/`.
- To retrain, use the notebook `student_exam_score_prediction.ipynb` and re-export the models.

### 4. Run the App
```bash
streamlit run streamlit_app.py
```
- The app will be available at `http://localhost:8501` by default.

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t student-score .
```

### 2. Run the Container
```bash
docker run -p 8501:8501 student-score
```
- Access the app at `http://localhost:8501`

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input student features (gender, race/ethnicity, parental education, lunch, test preparation).
3. Select a regression model.
4. Click **Predict Scores** to get the estimated exam results.

---

## 📊 Visualizations & EDA
- See `student_exam_score_prediction.ipynb` for:
  - Outlier analysis
  - Feature importance (tree models)
  - Correlation heatmap
  - Histograms, boxplots, pairplots, and more

---

## 📝 Model Details
- **Preprocessing**: StandardScaler, outlier detection, one-hot encoding for categorical features.
- **Models**: RandomForestRegressor, LinearRegression, GradientBoostingRegressor (with GridSearchCV for tuning).

---

## 📁 File Descriptions
- `streamlit_app.py`: Streamlit app, loads models, handles prediction and UI.
- `models/student_models.pkl`: Dictionary of trained regression models.
- `models/preprocessors.pkl`: Preprocessing tools and dropdown options.
- `environment.yml`: Conda environment dependencies.
- `Dockerfile`: Containerization instructions.
- `student_exam_score_prediction.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://kuldiiproject.com
- **Repository**: [GitHub - kuldii/student_score](https://github.com/kuldii/student_score)
- **Dataset**: [Kaggle Students Performance](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data)
- **UI**: [Streamlit](https://streamlit.io/)
- **ML**: [Scikit-learn](https://scikit-learn.org/)

---

For questions or contributions, please open an issue or pull request.
