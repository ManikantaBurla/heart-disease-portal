import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_lifestyle_model():
    print("--- Stage 1: Training Lifestyle Screening Model (70k Data) ---")
    
    # 1. Load the dataset using the correct semicolon separator
    try:
        df = pd.read_csv('data/cardio_train 70k.csv')
    except FileNotFoundError:
        print("ERROR: 'data/cardio_train 70k.csv' not found. Check your file name.")
        return

    print(f"Original 70k dataset shape: {df.shape}")

    # 2. Data Cleaning: Filter out impossible blood pressure outliers
    # Keeping Systolic (ap_hi) between 80 and 220, and Diastolic (ap_lo) between 40 and 130
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 130)]
    print(f"Shape after filtering blood pressure outliers: {df.shape}")

    # 3. Separate Features and Target
    # Dropping 'id' as it has no predictive value, and 'cardio' which is the target
    X = df.drop(columns=['id', 'cardio'], errors='ignore')
    y = df['cardio']

    # 4. Train/Test Split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test) * 100
    print(f"Stage 1 Test Accuracy: {accuracy:.2f}%")

    # 7. Save Model & Scaler assets
    joblib.dump(model, 'lifestyle_model.pkl')
    joblib.dump(scaler, 'lifestyle_scaler.pkl')
    print("Stage 1 assets successfully saved to workspace.\n")


def train_clinical_model():
    print("--- Stage 2: Training Deep Clinical Diagnostic Model (UCI Data) ---")
    
    # 1. Load the dataset (uses standard comma separator)
    try:
        df = pd.read_csv('data/heart_disease_cleveland.csv')
    except FileNotFoundError:
        print("ERROR: 'data/heart_disease_cleveland.csv' not found. Check your file name.")
        return

    print(f"UCI dataset shape: {df.shape}")

    # 2. Separate Features and Target ('target' is the standard column name)
    X = df.drop(columns=['target'])
    y = df['target']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test) * 100
    print(f"Stage 2 Test Accuracy: {accuracy:.2f}%")

    # 6. Save Model & Scaler assets
    joblib.dump(model, 'clinical_model.pkl')
    joblib.dump(scaler, 'clinical_scaler.pkl')
    print("Stage 2 assets successfully saved to workspace.\n")


if __name__ == "__main__":
    train_lifestyle_model()
    train_clinical_model()
    print("All backend models have been successfully compiled and saved!")