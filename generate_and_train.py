"""
Student Performance Predictor - Dataset Generation & Model Training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import os
import json

np.random.seed(42)
N = 2000

def generate_dataset():
    gender = np.random.choice(['Male', 'Female', 'Other'], N)
    study_hours = np.round(np.random.uniform(0, 12, N), 1)
    social_media_hours = np.round(np.random.uniform(0, 8, N), 1)
    attendance_pct = np.round(np.random.uniform(0, 100, N), 1)
    sleep_hours = np.round(np.random.uniform(4, 10, N), 1)
    parent_education = np.random.choice(['None', 'High School', 'Bachelor', 'Master', 'PhD'], N,
                                         p=[0.05, 0.35, 0.35, 0.18, 0.07])
    extra_curricular = np.random.choice(['None', 'Sports', 'Arts', 'Clubs', 'Multiple'], N,
                                         p=[0.2, 0.25, 0.2, 0.2, 0.15])
    part_time_job = np.random.choice(['Yes', 'No'], N, p=[0.3, 0.7])

    # Encode for score calculation
    gender_map = {'Male': 0, 'Female': 2, 'Other': 1}
    gender_num = np.array([gender_map[g] for g in gender], dtype=float)
    parent_edu_map = {'None': 0, 'High School': 2, 'Bachelor': 5, 'Master': 8, 'PhD': 10}
    parent_num = np.array([parent_edu_map[p] for p in parent_education])
    extra_map = {'None': 0, 'Sports': 3, 'Arts': 2, 'Clubs': 3, 'Multiple': 5}
    extra_num = np.array([extra_map[e] for e in extra_curricular])
    job_penalty = (part_time_job == 'Yes').astype(float) * -3

    sleep_quality = np.where((sleep_hours >= 7) & (sleep_hours <= 9), 5,
                             np.where((sleep_hours >= 6) & (sleep_hours < 7), 2, -3))

    base_score = (
        study_hours * 4.5
        + attendance_pct * 0.3
        + sleep_quality
        + parent_num * 0.8
        + extra_num * 0.5
        + gender_num * 0.5
        + job_penalty
        - social_media_hours * 2.2
        + np.random.normal(0, 5, N)
    )

    score = np.clip(base_score, 10, 100).round(1)

    df = pd.DataFrame({
        'gender': gender,
        'study_hours_per_day': study_hours,
        'social_media_hours_per_day': social_media_hours,
        'attendance_percentage': attendance_pct,
        'sleep_hours_per_day': sleep_hours,
        'parent_education_level': parent_education,
        'extra_curricular_activities': extra_curricular,
        'part_time_job': part_time_job,
        'score': score
    })
    return df


def train_model(df):
    le_gender = LabelEncoder()
    le_parent = LabelEncoder()
    le_extra = LabelEncoder()
    le_job = LabelEncoder()

    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    df['parent_edu_enc'] = le_parent.fit_transform(df['parent_education_level'])
    df['extra_enc'] = le_extra.fit_transform(df['extra_curricular_activities'])
    df['job_enc'] = le_job.fit_transform(df['part_time_job'])

    features = ['gender_enc', 'study_hours_per_day', 'social_media_hours_per_day',
                'attendance_percentage', 'sleep_hours_per_day',
                'parent_edu_enc', 'extra_enc', 'job_enc']

    X = df[features]
    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    print(f"CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    feature_importance = dict(zip(features, model.feature_importances_))

    encoders = {
        'gender': le_gender,
        'parent_education': le_parent,
        'extra_curricular': le_extra,
        'part_time_job': le_job
    }

    metrics = {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'cv_r2_mean': round(cv_scores.mean(), 4),
        'cv_r2_std': round(cv_scores.std(), 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_importance': {k: round(float(v), 4) for k, v in feature_importance.items()}
    }

    return model, encoders, metrics, features


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Generating dataset...")
    df = generate_dataset()
    df.to_csv('data/student_performance.csv', index=False)
    print(f"Dataset saved: {len(df)} records")
    print(df.describe())

    print("\nTraining model...")
    model, encoders, metrics, features = train_model(df)

    joblib.dump(model, 'models/model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(features, 'models/features.pkl')

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nModel & assets saved.")
    print(json.dumps(metrics, indent=2))
