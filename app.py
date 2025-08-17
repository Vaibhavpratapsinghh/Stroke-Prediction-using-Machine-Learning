import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# File path for all_assets.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(BASE_DIR, 'all_assets.pkl')

# Load saved model and preprocessing assets
with open(pickle_path, 'rb') as file:
    assets = pickle.load(file)

model = assets['stacked_model']
scaler = assets['scaler']
imputer = assets['imputer']
selected_features = assets['selected_features']
bmi_lower = assets['bmi_bounds']['lower']
bmi_upper = assets['bmi_bounds']['upper']

cat_cols = ['work_type', 'smoking_status']
label_cols = ['gender', 'Residence_type', 'ever_married']
columns_to_scale = ['age', 'bmi_capped', 'avg_glucose_level']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Safely get form inputs with request.form.get()
        data = {
            'gender': request.form.get('gender'),
            'age': request.form.get('age', type=int),
            'hypertension': request.form.get('hypertension', type=int),
            'heart_disease': request.form.get('heart_disease', type=int),
            'ever_married': request.form.get('ever_married'),
            'work_type': request.form.get('work_type'),
            'Residence_type': request.form.get('Residence_type'),
            'avg_glucose_level': request.form.get('avg_glucose_level', type=float),
            'bmi': request.form.get('bmi', type=float),
            'smoking_status': request.form.get('smoking_status')
        }

        # Check for missing inputs
        if None in data.values():
            return "Error: Please fill in all fields properly.", 400

        df_input = pd.DataFrame([data])

        # Replace 'Other' with 'Female' for gender
        df_input['gender'] = df_input['gender'].replace('Other', 'Female')

        # Impute bmi
        df_input[['bmi']] = imputer.transform(df_input[['bmi']])

        # Cap bmi
        df_input['bmi_capped'] = df_input['bmi'].clip(bmi_lower, bmi_upper)

        # One-hot encode cat_cols
        df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True, dtype=int)

        # Label encode label_cols
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in label_cols:
            df_encoded[col] = le.fit_transform(df_input[col])

        # Add missing columns if any from selected_features
        for col in selected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Scale numeric columns
        df_encoded[columns_to_scale] = scaler.transform(df_encoded[columns_to_scale])

        # Select only required features
        X_final = df_encoded[selected_features]

        # Predict
        prob = model.predict_proba(X_final)[0][1]
        prediction = "Yes" if prob >= 0.5 else "No"
        prob_percent = round(prob * 100, 2)

        return render_template('result.html', prediction=prediction, probability=prob_percent)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)