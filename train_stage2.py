import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("⏳ Downloading UCI Heart Disease Dataset...")
# Downloading the standard Kaggle/UCI dataset directly
url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
df = pd.read_csv(url)

# 1. Ensure columns perfectly match your app.py order
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[features]
y = df['target'] # 0 = Safe, 1 = Disease Detected

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale the data (Crucial for clinical metrics like Cholesterol)
print("⚖️ Scaling clinical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Model with BALANCED weights (This fixes the "lazy" guessing)
print("🧠 Training deep diagnostic Neural-style Forest...")
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    class_weight='balanced', # <-- THIS IS THE MAGIC FIX
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 5. Evaluate the model so you can see the true accuracy
y_pred = model.predict(X_test_scaled)
print("\n=== STAGE 2 MODEL EVALUATION ===")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("================================\n")

# 6. Save the assets for your Flask App
joblib.dump(model, 'clinical_model.pkl')
joblib.dump(scaler, 'clinical_scaler.pkl')

print("✅ SUCCESS: clinical_model.pkl and clinical_scaler.pkl have been updated!")