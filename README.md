# 🎓 Student Performance Predictor

A machine learning web application that predicts student exam scores based on study habits, lifestyle choices, and background factors. Built with Python, scikit-learn, and Streamlit.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [How to Use](#how-to-use)
8. [Model Performance](#model-performance)
9. [Tech Stack](#tech-stack)
10. [Understanding the Results](#understanding-the-results)

---

## 🔍 Project Overview

This project builds a complete end-to-end Artificial Intelligence and machine learning system to predict student academic performance (exam scores out of 100). The system:

- **Generates** a realistic synthetic dataset of 2,000 student records
- **Trains** a Gradient Boosting Regressor model with ~91% accuracy (R²)
- **Deploys** an interactive Streamlit web dashboard where anyone can:
  - Input their personal details and instantly get a predicted score
  - Explore the dataset with charts
  - Run a "What-If Simulator" to see how changing habits improves scores
  - Get personalised improvement tips

> **Why synthetic data?**
> Real student datasets often have privacy restrictions. We generate synthetic data using realistic mathematical relationships (e.g., more study hours → higher scores, excessive social media → lower scores) to simulate real-world patterns.

---

## ✨ Features

### 🔮 Prediction Tab
- Live predicted score (updates with sidebar inputs)
- Score gauge bar with colour-coded grade (A+ to F)
- Personalised improvement tips based on your inputs
- **What-If Simulator** — see how improving each habit changes your score

### 📊 Data Explorer Tab
- Score distribution histogram
- Score breakdown by gender
- Scatter plots: Study Hours vs Score, Attendance vs Score
- Average score by Parent Education Level
- Raw dataset preview

### 🧠 Model Insights Tab
- Model performance metrics: R², MAE, RMSE, Cross-Validation R²
- Feature importance bar chart (which factors matter most)
- Residual distribution (how far off predictions are)
- Actual vs Predicted scatter plot

---

## 📂 Dataset

### Input Features (8 total)

| Feature | Type | Range / Options | Impact |
|---|---|---|---|
| Gender | Categorical | Male / Female | Low |
| Study Hours Per Day | Numerical | 0 – 12 hours | Very High |
| Social Media Hours Per Day | Numerical | 0 – 8 hours | High (negative) |
| Attendance Percentage | Numerical | 00% – 100% | High |
| Sleep Hours Per Day | Numerical | 4 – 10 hours | Medium |
| Parent Education Level | Categorical | None / High School / Bachelor / Master / PhD | Medium |
| Extra-Curricular Activities | Categorical | None / Sports / Arts / Clubs / Multiple | Low |
| Part-Time Job | Categorical | Yes / No | Low-Medium |

### Target Variable
- **Score**: Continuous value from 0 to 100 (representing exam percentage)

### Score Generation Formula (simplified)
```
score = (study_hours × 4.5)
      + (attendance × 0.3)
      + sleep_quality_bonus
      + parent_education_bonus
      - (social_media × 2.2)
      ± random_noise
```

This formula reflects real-world research findings where study time is the strongest predictor, followed by attendance and reduced social media use.

---

## 🤖 Machine Learning Pipeline

### Step 1: Data Generation
```
python generate_and_train.py
```
- Creates 2,000 synthetic student records with realistic correlations
- Saves to `data/student_performance.csv`

### Step 2: Preprocessing
- **Label Encoding** for categorical columns (Gender, Parent Education, etc.)
- No scaling needed for tree-based models
- 80/20 train-test split with fixed random seed for reproducibility

### Step 3: Model Selection
We chose **Gradient Boosting Regressor** because:
- Handles mixed feature types (numerical + encoded categorical) well
- Robust to outliers
- Captures non-linear relationships (e.g., sleep: too little AND too much hurts performance)
- Generally outperforms Linear Regression and basic Decision Trees

Hyperparameters used:
```python
GradientBoostingRegressor(
    n_estimators=300,   # 300 trees
    learning_rate=0.05, # Small steps for better generalisation
    max_depth=5,        # Moderate complexity
    min_samples_split=10
)
```

### Step 4: Evaluation
| Metric | Value | What it means |
|---|---|---|
| R² Score | ~0.90 | Model explains 90% of variance in scores |
| MAE | ~4.6 | On average, predictions are off by only 4.6 points |
| RMSE | ~5.9 | Root mean squared error (penalises large errors more) |
| CV R² (5-fold) | ~0.91 | Consistent performance across different data splits |

### Step 5: Saving the Model
Model artifacts saved to the `models/` directory:
- `model.pkl` — trained Gradient Boosting model
- `encoders.pkl` — fitted Label Encoders for each categorical feature
- `features.pkl` — ordered list of feature names
- `metrics.json` — performance metrics

---

## 📁 Project Structure

```
student_performance_predictor/
│
├── app.py                    # Main Streamlit web application
├── generate_and_train.py     # Dataset generation + model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   └── student_performance.csv   # Generated dataset (2000 records)
│
└── models/
    ├── model.pkl             # Trained ML model
    ├── encoders.pkl          # Label encoders
    ├── features.pkl          # Feature list
    └── metrics.json          # Model performance metrics
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Extract the Project
```bash
cd student_performance_predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Data & Train Model
```bash
python generate_and_train.py
```
This creates the `data/` and `models/` directories automatically.

### Step 4: Launch the Web App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖥️ How to Use

1. **Open the app** — The left sidebar contains all input controls
2. **Fill in your details**:
   - Adjust sliders for study hours, social media use, attendance, sleep
   - Select gender, parent education, extra-curriculars, and job status
3. **View your prediction** — The "Prediction" tab shows your score instantly
4. **Read your tips** — Personalised recommendations appear below the score
5. **Try the simulator** — See what happens if you study 1 more hour, or sleep better
6. **Explore the data** — Switch to the "Data Explorer" tab for visual insights
7. **Understand the model** — The "Model Insights" tab explains how the AI works

---

## 📊 Model Performance

```
Algorithm:      Gradient Boosting Regressor
Training Size:  1,600 students
Test Size:      400 students
R² Score:       0.8970
MAE:            4.62 points
RMSE:           5.86 points
CV R² (5-fold): 0.9106 ± 0.006
```

**Feature Importance (top factors):**
1. 📖 Study Hours per Day — ~76% importance (by far the strongest predictor)
2. 🏫 Attendance Percentage — ~9.5% importance
3. 📱 Social Media Hours — ~8.9% importance
4. 😴 Sleep Hours — ~3.8% importance
5. Others (parent education, extra-curricular, job, gender) — < 3% each

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Serialisation | joblib |

---

## 🎯 Understanding the Results

### Grade Scale
| Score | Grade | Meaning |
|---|---|---|
| 90–100 | A+ | Outstanding |
| 80–89 | A | Excellent |
| 70–79 | B+ | Very Good |
| 60–69 | B | Good |
| 50–59 | C | Average |
| 40–49 | D | Below Average |
| < 40 | F | Needs Improvement |

### Key Insights from the Data
- Students who study **7+ hours/day** score on average **20+ points higher** than those studying < 3 hours
- Reducing social media from 6h to 2h per day correlates with a **~9 point increase** in score
- Students with **≥ 85% attendance** consistently outperform those with < 70% attendance
- Optimal sleep (7–8 hours) outperforms both under-sleeping (< 6h) and over-sleeping (> 9h)

---

## ⚠️ Limitations & Disclaimer

- This app uses **synthetic data** — predictions are for educational/demo purposes only
- Real student performance depends on many more factors (teacher quality, mental health, learning disabilities, motivation, etc.)
- The model is a simplified representation and should **not** be used for actual academic assessment

---

## 🔮 Future Improvements

- [ ] Upload real anonymised datasets for training
- [ ] Add subject-wise score prediction (Maths, Science, English)
- [ ] Include mental health / stress level as an input
- [ ] Export prediction report as PDF
- [ ] Add batch prediction (upload CSV of multiple students)
- [ ] Hyperparameter tuning with GridSearchCV

---

*Built with ❤️ using Python & Streamlit*
