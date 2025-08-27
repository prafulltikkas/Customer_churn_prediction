from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Initialize model as None
model = None

# Try to load the model
try:
    # Try different possible model paths
    model_paths = [
        'models/selectedCatBoostClassifier.pkl',
        'selectedCatBoostClassifier.pkl',
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Model loaded successfully from {path}!")
            break
    
    if model is None:
        print("Error: Could not find the model file. Please ensure it exists in one of these locations:")
        for path in model_paths:
            print(f"  - {path}")
            
except Exception as e:
    print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return render_template('index.html', error="Model not loaded. Please contact administrator.")
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert to appropriate data types and prepare for prediction
        features = [
            float(data['total_logins']),
            float(data['tickets_raised']),
            float(data['customer_tenure']),
            float(data['sentiment_score']),
            int(data['onboarding_year']),
            float(data['loans_accessed']),
            float(data['loans_taken']),
            float(data['monthly_avg_balance'])
        ]
        
        # Make prediction
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        
        # Calculate risk percentage
        churn_probability = probability[0][1] * 100
        
        # Determine risk level
        if churn_probability < 30:
            risk_level = "Low Risk"
            risk_color = "green"
        elif churn_probability < 70:
            risk_level = "Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        result = {
            'prediction': 'Will Churn' if prediction[0] == 1 else 'Will Not Churn',
            'probability': round(churn_probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color
        }
        
        return render_template('index.html', prediction=result, form_data=data)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)