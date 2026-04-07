"""
Student Performance Predictor - Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f0f1a; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1a1a2e !important; }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label { color: #e0e0ff !important; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a, #252545);
        border: 1px solid #3a3a6a;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(100,100,255,0.08);
    }
    .metric-card h3 { color: #aaaacc; font-size: 0.85rem; margin: 0 0 6px 0; letter-spacing: 1px; text-transform: uppercase; }
    .metric-card p  { color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0; }

    /* Score display */
    .score-box {
        background: linear-gradient(135deg, #1a1a4a, #2a1a5a);
        border: 2px solid #6464ff;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 30px rgba(100,100,255,0.25);
    }
    .score-box h1 { font-size: 4rem; color: #ffffff; margin: 0; }
    .score-box h3 { color: #aaaaee; margin: 0; font-size: 1.1rem; }
    .score-box .grade { font-size: 2.2rem; font-weight: 800; margin-top: 8px; }

    /* Tips */
    .tip-box {
        background: #1c1c38;
        border-left: 4px solid #6464ff;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 8px 0;
        color: #ccccee;
        font-size: 0.92rem;
    }

    /* Headers */
    h1, h2, h3 { color: #e0e0ff !important; }
    .stMarkdown p { color: #ccccdd; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #aaaacc; }
    .stTabs [aria-selected="true"] { color: #8080ff !important; }

    /* Divider */
    hr { border-color: #2a2a4a; }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #c0c0ff;
        border-bottom: 2px solid #3a3a6a;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model Artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("models/model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    features = joblib.load("models/features.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return model, encoders, features, metrics

@st.cache_data
def load_data():
    df = pd.read_csv("data/student_performance.csv")
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

model, encoders, features, metrics = load_artifacts()
df = load_data()

# ─── Helper: Grade + Colour ──────────────────────────────────────────────────
def get_grade_info(score):
    if score >= 90:   return "A+", "#00e676", "Outstanding 🏆"
    elif score >= 80: return "A",  "#69f0ae", "Excellent 🌟"
    elif score >= 70: return "B+", "#40c4ff", "Very Good 👏"
    elif score >= 60: return "B",  "#7986cb", "Good 👍"
    elif score >= 50: return "C",  "#ffca28", "Average 📚"
    elif score >= 40: return "D",  "#ffa726", "Below Average ⚠️"
    else:              return "F",  "#ef5350", "Needs Improvement 🔴"

def safe_label_encode(encoder, value):
    """Encode a single value safely; falls back to 0 if label is unseen or None."""
    try:
        val = str(value).strip() if value is not None else encoder.classes_[0]
        if val not in encoder.classes_:
            val = encoder.classes_[0]
        return int(encoder.transform([val])[0])
    except Exception:
        return 0

def predict_score(inputs: dict) -> float:
    row = {
        'gender_enc':                  safe_label_encode(encoders['gender'],           inputs['gender']),
        'study_hours_per_day':         inputs['study_hours'],
        'social_media_hours_per_day':  inputs['social_media'],
        'attendance_percentage':       inputs['attendance'],
        'sleep_hours_per_day':         inputs['sleep_hours'],
        'parent_edu_enc':              safe_label_encode(encoders['parent_education'], inputs['parent_edu']),
        'extra_enc':                   safe_label_encode(encoders['extra_curricular'], inputs['extra_curr']),
        'job_enc':                     safe_label_encode(encoders['part_time_job'],    inputs['part_time_job']),
    }
    X = pd.DataFrame([row])[features]
    return float(np.clip(model.predict(X)[0], 0, 100))

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Info")
    st.markdown("---")

    gender      = st.selectbox("👤 Gender",           ["Male", "Female", "Other"])
    study_hours = st.slider("📖 Study Hours / Day",   0.0, 12.0, 5.0, 0.5)
    social_med  = st.slider("📱 Social Media Hrs/Day", 0.0, 8.0,  2.0, 0.5)
    attendance  = st.slider("🏫 Attendance %",         00.0, 100.0, 50.0, 1.0)
    sleep_hrs   = st.slider("😴 Sleep Hours / Day",   4.0, 10.0, 7.0, 0.5)
    st.markdown("---")
    st.markdown("#### 📋 Background")
    parent_edu  = st.selectbox("🎓 Parent Education",   ["None", "High School", "Bachelor", "Master", "PhD"])
    extra_curr  = st.selectbox("⚽ Extra-Curricular",   ["None", "Sports", "Arts", "Clubs", "Multiple"])
    part_job    = st.selectbox("💼 Part-Time Job",       ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Score", use_container_width=True, type="primary")

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Performance Predictor")
st.markdown("*Predict exam scores using study habits, lifestyle & background factors.*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Explorer", "🧠 Model Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    inputs = {
        'gender': gender, 'study_hours': study_hours, 'social_media': social_med,
        'attendance': attendance, 'sleep_hours': sleep_hrs,
        'parent_edu': parent_edu, 'extra_curr': extra_curr, 'part_time_job': part_job
    }

    if predict_btn or True:          # always show a live preview
        score = predict_score(inputs)
        grade, color, label = get_grade_info(score)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown(f"""
            <div class="score-box">
                <h3>Predicted Score</h3>
                <h1>{score:.1f}<span style="font-size:1.6rem;color:#aaaaee">/100</span></h1>
                <div class="grade" style="color:{color}">{grade} — {label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 0.7))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#1a1a2e')
            ax.barh(0, 100, color='#2a2a4a', height=0.55)
            ax.barh(0, score, color=color, height=0.55)
            ax.set_xlim(0, 100)
            ax.set_yticks([]); ax.set_xticks(range(0, 101, 10))
            ax.tick_params(colors='#aaaacc', labelsize=8)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_xlabel("Score", color='#aaaacc', fontsize=8)
            plt.tight_layout(pad=0.3)
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### 📋 Input Summary")
            summary = {
                "Gender": gender, "Study Hrs": f"{study_hours}h",
                "Social Media": f"{social_med}h", "Attendance": f"{attendance}%",
                "Sleep": f"{sleep_hrs}h", "Parent Edu": parent_edu,
                "Extra-Curr": extra_curr, "Part-time Job": part_job
            }
            for k, v in summary.items():
                st.markdown(f"**{k}:** {v}")

        st.markdown("---")

        # ── Personalised Tips ───────────────────────────────────────────────
        st.markdown("#### 💡 Personalised Recommendations")
        tips = []
        if study_hours < 4:
            tips.append("📖 Increase study hours — even 1–2 more hours daily can significantly raise your score.")
        if social_med > 4:
            tips.append("📵 Reduce social media use. Try the Pomodoro technique: 25 min study, 5 min break.")
        if attendance < 75:
            tips.append("🏫 Attendance is low. Regular classes improve understanding and retention by 30–40%.")
        if sleep_hrs < 6:
            tips.append("😴 Poor sleep hurts memory consolidation. Aim for 7–8 hours per night.")
        if sleep_hrs > 9:
            tips.append("⏰ Oversleeping can reduce focus. 7–8 hours is the sweet spot for academic performance.")
        if part_job == "Yes":
            tips.append("💼 Part-time work adds stress. Try to keep it under 15 hrs/week during exams.")
        if study_hours >= 7 and attendance >= 85:
            tips.append("🌟 Great dedication! Keep consistent study sessions and review notes regularly.")
        if not tips:
            tips.append("✅ Your habits look solid! Maintain consistency and you're on track for great results.")

        for t in tips:
            st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

        # ── What-if Simulator ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 What-If Simulator")
        st.markdown("*See how improving one habit changes your predicted score.*")

        scenarios = {
            "Current":             inputs.copy(),
            "+1h Study":           {**inputs, 'study_hours': min(12, study_hours + 1)},
            "-1h Social Media":    {**inputs, 'social_media': max(0, social_med - 1)},
            "Attendance → 95%":    {**inputs, 'attendance': 95},
            "Sleep → 8h":          {**inputs, 'sleep_hours': 8},
            "All Improved":        {**inputs, 'study_hours': min(12, study_hours+2),
                                              'social_media': max(0, social_med-1.5),
                                              'attendance': max(attendance, 90),
                                              'sleep_hours': 8},
        }

        sc_names, sc_scores, sc_colors = [], [], []
        for name, sc in scenarios.items():
            s = predict_score(sc)
            sc_names.append(name); sc_scores.append(round(s, 1))
            _, c, _ = get_grade_info(s)
            sc_colors.append(c)

        fig2, ax2 = plt.subplots(figsize=(9, 3.2))
        fig2.patch.set_facecolor('#0f0f1a'); ax2.set_facecolor('#1a1a2e')
        bars = ax2.barh(sc_names, sc_scores, color=sc_colors, edgecolor='#2a2a4a', height=0.55)
        ax2.set_xlim(0, 100)
        for bar, val in zip(bars, sc_scores):
            ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val}", va='center', color='white', fontsize=9)
        ax2.tick_params(colors='#ccccee'); ax2.set_xlabel("Score", color='#aaaacc')
        for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a4a')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    stats = [("Total Students", len(df)), ("Avg Score", f"{df.score.mean():.1f}"),
             ("Median Score", f"{df.score.median():.1f}"), ("Std Dev", f"{df.score.std():.1f}")]
    for col, (label, val) in zip([c1,c2,c3,c4], stats):
        col.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{val}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Score Distribution**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        ax.hist(df.score, bins=30, color='#6464ff', edgecolor='#1a1a2e', alpha=0.85)
        ax.axvline(df.score.mean(), color='#ff8080', lw=2, ls='--', label=f'Mean={df.score.mean():.1f}')
        ax.set_xlabel("Score", color='#aaaacc'); ax.set_ylabel("Count", color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Score by Gender**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        colors_g = ['#6464ff', '#ff64aa', '#64ffda']
        for i, (g, grp) in enumerate(df.groupby('gender')):
            ax.hist(grp.score, bins=20, alpha=0.7, color=colors_g[i], label=g, edgecolor='#1a1a2e')
        ax.set_xlabel("Score", color='#aaaacc'); ax.set_ylabel("Count", color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Study Hours vs Score**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        sc = ax.scatter(df.study_hours_per_day, df.score, c=df.score, cmap='plasma', alpha=0.4, s=10)
        plt.colorbar(sc, ax=ax, label='Score').ax.yaxis.label.set_color('#aaaacc')
        ax.set_xlabel("Study Hours", color='#aaaacc'); ax.set_ylabel("Score", color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("**Attendance vs Score**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        sc = ax.scatter(df.attendance_percentage, df.score, c=df.score, cmap='cool', alpha=0.4, s=10)
        plt.colorbar(sc, ax=ax, label='Score').ax.yaxis.label.set_color('#aaaacc')
        ax.set_xlabel("Attendance %", color='#aaaacc'); ax.set_ylabel("Score", color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Avg Score by Parent Education Level**")
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
    order = ['None', 'High School', 'Bachelor', 'Master', 'PhD']
    means = df.groupby('parent_education_level')['score'].mean().reindex(order)
    bars = ax.bar(means.index, means.values, color='#8080ff', edgecolor='#1a1a2e')
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.5, f"{val:.1f}", ha='center', va='bottom',
                color='white', fontsize=9)
    ax.set_ylabel("Avg Score", color='#aaaacc'); ax.tick_params(colors='#aaaacc')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    if st.checkbox("🔍 Show raw dataset sample"):
        st.dataframe(df.sample(20, random_state=1).reset_index(drop=True),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🧠 Model Performance & Insights</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    kpis = [("R² Score", f"{metrics['r2']:.4f}"),
            ("MAE", f"{metrics['mae']}"),
            ("RMSE", f"{metrics['rmse']}"),
            ("CV R² (5-fold)", f"{metrics['cv_r2_mean']:.4f}")]
    for col, (lbl, val) in zip([m1,m2,m3,m4], kpis):
        col.markdown(f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Importance**")
        fi = metrics['feature_importance']
        labels_map = {
            'study_hours_per_day': 'Study Hours',
            'attendance_percentage': 'Attendance %',
            'social_media_hours_per_day': 'Social Media',
            'sleep_hours_per_day': 'Sleep Hours',
            'parent_edu_enc': 'Parent Education',
            'job_enc': 'Part-time Job',
            'extra_enc': 'Extra-Curricular',
            'gender_enc': 'Gender'
        }
        fi_sorted = sorted(fi.items(), key=lambda x: x[1])
        names  = [labels_map.get(k, k) for k, _ in fi_sorted]
        values = [v for _, v in fi_sorted]
        cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(names)))

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        bars = ax.barh(names, values, color=cmap, edgecolor='#1a1a2e')
        for bar, val in zip(bars, values):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
                    va='center', color='white', fontsize=8)
        ax.set_xlabel("Importance", color='#aaaacc'); ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Score Prediction Residuals**")

        def safe_encode(le, series):
            """Encode a Series using a fitted LabelEncoder, replacing unseen/NaN with 0."""
            known = set(le.classes_)
            return series.fillna(le.classes_[0]).apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )

        le_g = encoders['gender']; le_p = encoders['parent_education']
        le_e = encoders['extra_curricular']; le_j = encoders['part_time_job']
        df2 = df.copy()
        df2['gender_enc']       = safe_encode(le_g, df2['gender'])
        df2['parent_edu_enc']   = safe_encode(le_p, df2['parent_education_level'])
        df2['extra_enc']        = safe_encode(le_e, df2['extra_curricular_activities'])
        df2['job_enc']          = safe_encode(le_j, df2['part_time_job'])
        # Drop any rows where numerical columns have NaN
        df2 = df2.dropna(subset=features)
        X_all = df2[features]
        y_pred_all = model.predict(X_all)
        residuals = df2['score'].values - y_pred_all

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
        ax.hist(residuals, bins=40, color='#40c4ff', edgecolor='#1a1a2e', alpha=0.85)
        ax.axvline(0, color='#ff8080', lw=2, ls='--', label='Zero error')
        ax.set_xlabel("Residual (Actual − Predicted)", color='#aaaacc')
        ax.set_ylabel("Count", color='#aaaacc'); ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Actual vs Predicted Scores (sample of 300)**")
    sample_idx = np.random.choice(len(df2), 300, replace=False)
    actual_s   = df2['score'].values[sample_idx]
    pred_s     = y_pred_all[sample_idx]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#1a1a2e')
    ax.scatter(actual_s, pred_s, alpha=0.5, color='#8080ff', s=18)
    ax.plot([10, 100], [10, 100], color='#ff8080', lw=1.5, ls='--', label='Perfect prediction')
    ax.set_xlabel("Actual Score", color='#aaaacc'); ax.set_ylabel("Predicted Score", color='#aaaacc')
    ax.tick_params(colors='#aaaacc')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("""
    #### 🤖 About the Model
    | Property | Value |
    |---|---|
    | Algorithm | Gradient Boosting Regressor |
    | Dataset Size | 2,000 students |
    | Features | 8 input features |
    | Train / Test Split | 80% / 20% |
    | Cross-Validation | 5-Fold |
    | Best CV R² | ~0.91 |
    """)
