from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# --- 1. Load Machine Learning Assets ---
def load_models():
    try:
        lifestyle_model = joblib.load('lifestyle_model.pkl')
        lifestyle_scaler = joblib.load('lifestyle_scaler.pkl')
        clinical_model = joblib.load('clinical_model.pkl')
        clinical_scaler = joblib.load('clinical_scaler.pkl')
        print("✅ All Machine Learning assets loaded successfully.")
        return lifestyle_model, lifestyle_scaler, clinical_model, clinical_scaler
    except Exception as e:
        print(f"⚠️ WARNING: Could not load models. Error: {e}")
        return None, None, None, None

lifestyle_model, lifestyle_scaler, clinical_model, clinical_scaler = load_models()

# --- 2. Gateway Routes ---

@app.route('/')
def home():
    """Renders the Gateway Dashboard (index.html)"""
    return render_template('index.html')

# --- 3. The 'Common Man' Patient Flow ---

@app.route('/commonman')
def commonman_mode():
    """Renders the simplified symptom-based form."""
    return render_template('commonman.html')

@app.route('/commonman_screen', methods=['POST'])
def commonman_screen():
    """Processes Common Man Mode data using Deterministic Imputation."""
    if not lifestyle_model:
        return "Models not loaded.", 500

    try:
        # Extract known variables
        age_years = float(request.form.get('age'))
        age_days = age_years * 365.25 
        gender = int(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        smoke = int(request.form.get('smoke'))
        alco = int(request.form.get('alco'))
        active = int(request.form.get('active'))

        # Extract proxy symptoms
        has_headache = int(request.form.get('headache'))
        has_thirst = int(request.form.get('thirst'))

        # --- FIX: Safe Height Conversion ---
        # If a user types 1.7 (meters) instead of 170 (cm), this automatically fixes it
        if height < 10.0:
            height = height * 100

        # --- REFINED DETERMINISTIC IMPUTATION LOGIC ---
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        # 1. Blood Pressure: Only flag high BP if they ACTUALLY have symptoms
        if has_headache == 1:
            ap_hi, ap_lo = 150, 95  # Hypertension proxy
        else:
            ap_hi, ap_lo = 120, 80  # Safe, healthy baseline

        # 2. Glucose: Only flag diabetes if they have the thirst symptom
        if has_thirst == 1:
            gluc = 3
        else:
            gluc = 1

        # 3. Cholesterol: Only penalize if BMI is significantly high (Obese)
        if bmi > 30 and age_years > 50:
            cholesterol = 3
        elif bmi > 28:
            cholesterol = 2
        else:
            cholesterol = 1
        # ----------------------------------------------

        # Create numpy array with the imputed values
        features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        
        # Scale and predict
        features_scaled = lifestyle_scaler.transform(features)
        risk_probability = lifestyle_model.predict_proba(features_scaled)[0][1] * 100
        
        # In Village/Common Man mode, we NEVER escalate to Stage 2 (clinical.html).
        if risk_probability >= 50.0:
            return render_template('result.html', escalated=True, prediction=1, confidence=f"{risk_probability:.1f}")
        else:
            return render_template('result.html', escalated=False, risk=f"{risk_probability:.1f}")

    except Exception as e:
        return f"Error processing Common Man data: {e}", 400

# --- 4. The Medical Professional Flow ---

@app.route('/professional')
def professional_mode():
    """Renders the Stage 1 clinical triage for doctors."""
    return render_template('triage.html')

@app.route('/triage_screen', methods=['POST'])
def triage_screen():
    """Processes Stage 1 data specifically for Medical Professionals."""
    if not lifestyle_model:
        return "Models not loaded.", 500

    try:
        # Extract explicit medical values entered by the doctor
        age_years = float(request.form.get('age'))
        age_days = age_years * 365.25 
        gender = int(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        ap_hi = float(request.form.get('ap_hi'))
        ap_lo = float(request.form.get('ap_lo'))
        cholesterol = int(request.form.get('cholesterol'))
        gluc = int(request.form.get('gluc'))
        smoke = int(request.form.get('smoke'))
        alco = int(request.form.get('alco'))
        active = int(request.form.get('active'))

        features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        features_scaled = lifestyle_scaler.transform(features)
        
        risk_probability = lifestyle_model.predict_proba(features_scaled)[0][1] * 100
        
        # THE ESCALATION LOGIC
        if risk_probability >= 50.0:
            # Stage 1 test shows high risk -> automatically push to Stage 2
            # The UCI dataset uses 1=Male, 0=Female, so we convert the 70k gender format
            uci_sex = 1 if gender == 2 else 0 
            
            return render_template('clinical.html', 
                                   escalated=True, 
                                   prev_risk=f"{risk_probability:.1f}",
                                   age=int(age_years),
                                   sex=uci_sex)
        else:
            # Patient is healthy -> show the Safe Result page immediately
            return render_template('result.html', escalated=False, risk=f"{risk_probability:.1f}")

    except Exception as e:
        return f"Error processing Triage data: {e}", 400

@app.route('/diagnose', methods=['POST'])
def diagnose_patient():
    """Processes Stage 2 data for final clinical diagnosis."""
    if not clinical_model:
        return "Models not loaded.", 500

    try:
        # Extract the 13 clinical features from the HTML form
        input_data = [
            float(request.form.get('age')), float(request.form.get('sex')), float(request.form.get('cp')),
            float(request.form.get('trestbps')), float(request.form.get('chol')), float(request.form.get('fbs')),
            float(request.form.get('restecg')), float(request.form.get('thalach')), float(request.form.get('exang')),
            float(request.form.get('oldpeak')), float(request.form.get('slope')), float(request.form.get('ca')),
            float(request.form.get('thal'))
        ]
        
        # Format, scale, and predict
        features = np.array([input_data])
        features_scaled = clinical_scaler.transform(features)
        
        # Get Kaggle's backward prediction
        raw_prediction = clinical_model.predict(features_scaled)[0]
        probabilities = clinical_model.predict_proba(features_scaled)[0]
        
        # --- THE TRANSLATION FLIP ---
        if raw_prediction == 0:
            # Kaggle thinks 0 is Disease. We send 1 to trigger Amber HTML.
            prediction_for_html = 1
            confidence = probabilities[0] * 100
        else:
            # Kaggle thinks 1 is Safe. We send 0 to trigger Green HTML.
            prediction_for_html = 0
            confidence = probabilities[1] * 100

        # === THE WIRETAP ===
        print("\n" + "="*35)
        print("🕵️‍♂️ STAGE 2 DIAGNOSTIC WIRETAP")
        print(f"Raw Input: {input_data}")
        print(f"Kaggle Output: {raw_prediction} | Confidence: {confidence:.1f}%")
        print(f"Sent to HTML: {prediction_for_html}")
        print("="*35 + "\n")

        return render_template('result.html', escalated=True, prediction=prediction_for_html, confidence=f"{confidence:.1f}")

    except Exception as e:
        return f"Error processing Stage 2 data: {e}", 400
    
# --- 5. Run Server ---
if __name__ == '__main__':
    app.run(debug=True)