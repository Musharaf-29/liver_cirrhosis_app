from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Paths to model and normalizer files
MODEL_PATH = 'models/rf_acc_68.pkl'
NORMALIZER_PATH = 'models/normalizer.pkl'
DATASET_PATH = 'cirrhosis.csv'

# Function to train and save the model
def train_and_save_model():
    print("Training and saving model...")
    try:
        data = pd.read_csv(DATASET_PATH)

        # Add missing columns with dummy values
        if 'Direct_Bilirubin' not in data.columns:
            data['Direct_Bilirubin'] = data['Bilirubin'] * 0.3
        if 'SGPT' not in data.columns:
            data['SGPT'] = data['SGOT']
        if 'Total_Protiens' not in data.columns:
            data['Total_Protiens'] = data['Albumin'] + 2
        if 'Albumin_and_Globulin_Ratio' not in data.columns:
            data['Albumin_and_Globulin_Ratio'] = data['Albumin'] / 2

        # Define features and target
        features = [
            'Bilirubin',
            'Direct_Bilirubin',
            'Alk_Phos',
            'SGPT',
            'SGOT',
            'Total_Protiens',
            'Albumin',
            'Albumin_and_Globulin_Ratio',
            'Age'
        ]
        target = 'Stage'

        # Check for missing required columns
        if not all(col in data.columns for col in features + [target]):
            print("Error: Required columns missing. Available:", data.columns.tolist())
            return

        X = data[features]
        y = data[target]

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.dropna()
        X = X.loc[y.index]

        print(f"Training with {X.shape[1]} features: {features}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize
        normalizer = StandardScaler()
        X_train_scaled = normalizer.fit_transform(X_train)

        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score
        X_test_scaled = normalizer.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(NORMALIZER_PATH, 'wb') as f:
            pickle.dump(normalizer, f)

        print("Model and normalizer saved successfully.")

    except Exception as e:
        print("Training failed:", str(e))

# Load model and scaler
def load_model_and_normalizer():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(NORMALIZER_PATH) or os.path.getsize(MODEL_PATH) == 0:
            print("Model or normalizer file missing. Training new model...")
            train_and_save_model()

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(NORMALIZER_PATH, 'rb') as f:
            normalizer = pickle.load(f)
        return model, normalizer
    except Exception as e:
        print("Error loading model:", e)
        train_and_save_model()
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(NORMALIZER_PATH, 'rb') as f:
            normalizer = pickle.load(f)
        return model, normalizer

# Load at startup
model, normalizer = load_model_and_normalizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        data = [
            float(request.form['bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alk_phos']),
            float(request.form['sgpt']),
            float(request.form['sgot']),
            float(request.form['total_proteins']),
            float(request.form['albumin']),
            float(request.form['ag_ratio']),
            float(request.form['age'])
        ]
        print("Input features:", data)

        # Normalize
        data_scaled = normalizer.transform([data])

        # Predict
        prediction = model.predict(data_scaled)[0]
        stages = {
            1: "Stage 1 (Mild)",
            2: "Stage 2 (Moderate)",
            3: "Stage 3 (Severe)",
            4: "Stage 4 (End-Stage)"
        }
        result = stages.get(prediction, f"Stage {prediction} (Unknown)")
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/inner-page')
def inner_page():
    return render_template('inner-page.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

if __name__ == '__main__':
    app.run(debug=True)
