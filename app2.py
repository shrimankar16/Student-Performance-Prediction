import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")  # Fix 1: was warnings.filter()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Card container */
.main-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    padding: 2.5rem 2.5rem 2rem;
    backdrop-filter: blur(20px);
    margin-top: 1rem;
}

/* Title */
.title-block {
    text-align: center;
    margin-bottom: 2rem;
}
.title-block h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: rgba(255,255,255,0.5);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
}

/* Sliders */
.stSlider > label {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500;
    font-size: 0.93rem;
}
.stSlider [data-baseweb="slider"] {
    margin-top: 0.2rem;
}
.stSlider [data-testid="stThumbValue"] {
    background: #a78bfa !important;
    color: white !important;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}

/* Selectbox */
.stSelectbox > label {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500;
    font-size: 0.93rem;
}
.stSelectbox [data-baseweb="select"] {
    background-color: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    margin-top: 1.5rem;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
}

/* Success / result box */
.stSuccess {
    background: rgba(52, 211, 153, 0.12) !important;
    border: 1px solid rgba(52, 211, 153, 0.35) !important;
    border-radius: 16px !important;
    color: #34d399 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-align: center;
    margin-top: 1rem;
}

/* Score meter */
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 10px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    transition: width 0.6s ease;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.6rem 0 !important;
}

/* Hide default Streamlit footer */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except Exception:
        return None

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🎓 Score Predictor</h1>
    <p>Enter your study habits below to predict your exam performance</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Study Hours per Day", 0.0, 12.0, 2.0, step=0.5)
    attendance  = st.slider("🏫 Attendance Percentage", 0.0, 100.0, 80.0, step=1.0)
    sleep_hours = st.slider("😴 Sleep Hours per Night", 0.0, 12.0, 7.0, step=0.5)  # Fix 2: was 12:0

with col2:
    mental_health = st.slider("🧠 Mental Health Rating (1–10)", 1, 10, 5)
    part_time_job = st.selectbox("💼 Part-Time Job", ["No", "Yes"])  # Fix 3: was st.select()

    # Quick stats summary
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04);border-radius:14px;padding:1rem 1.2rem;border:1px solid rgba(255,255,255,0.09);'>
        <div style='color:rgba(255,255,255,0.4);font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.7rem;font-family:Syne,sans-serif;'>Quick Summary</div>
        <div style='color:#a78bfa;font-size:0.85rem;'>Study: <b style="color:white">{study_hours}h/day</b></div>
        <div style='color:#60a5fa;font-size:0.85rem;'>Sleep: <b style="color:white">{sleep_hours}h/night</b></div>
        <div style='color:#34d399;font-size:0.85rem;'>Attendance: <b style="color:white">{attendance:.0f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("⚡ Predict My Exam Score"):
    if model is None:
        st.error("⚠️ Model file `best_model.pkl` not found. Make sure it's in the same folder as this script.")
    else:
        input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))

        grade = (
            "A+" if prediction >= 90 else
            "A"  if prediction >= 80 else
            "B"  if prediction >= 70 else
            "C"  if prediction >= 60 else
            "D"  if prediction >= 50 else "F"
        )

        st.success(f"🎯 Predicted Exam Score: **{prediction:.1f} / 100** — Grade **{grade}**")

        # Score progress bar
        st.markdown(f"""
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{prediction}%;"></div>
        </div>
        <div style='display:flex;justify-content:space-between;color:rgba(255,255,255,0.3);font-size:0.75rem;margin-top:0.3rem;'>
            <span>0</span><span>50</span><span>100</span>
        </div>
        """, unsafe_allow_html=True)

        # Personalised tip
        st.markdown("<br>", unsafe_allow_html=True)
        if study_hours < 3:
            st.info("💡 **Tip:** Try increasing your daily study time to 3–5 hours for a significant score boost.")
        elif attendance < 70:
            st.info("💡 **Tip:** Higher attendance strongly correlates with better performance. Aim for 80%+.")
        elif sleep_hours < 6:
            st.info("💡 **Tip:** You may be sleep-deprived. 7–8 hours of sleep improves retention and focus.")
        else:
            st.info("✅ **Great habits!** Keep your current routine and stay consistent before the exam.")