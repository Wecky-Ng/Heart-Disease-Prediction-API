import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
# Removed database imports: mysql.connector, Error
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Initialize CORS, allowing requests from your Vercel frontend
CORS(app, resources={r"/predict": {"origins": "https://heart-disease-ai-fyp.vercel.app"}})

# --- Removed Database Configuration ---

# --- Model Loading ---
MODEL_PATH = 'tuned_ensemble_cat_lgbm_v5.pkl' # Ensure this model file is in the same directory or provide the correct path
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Ensure the file exists.")
    # Depending on requirements, you might want to exit or handle this differently
    exit(1) # Exit if model can't be loaded
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# --- Helper Function for Age Category --- 
def get_age_category(age):
    if not isinstance(age, (int, float)) or age < 18:
        return 'Unknown' # Or handle error appropriately
    if 18 <= age <= 24:
        return '18-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 34:
        return '30-34'
    elif 35 <= age <= 39:
        return '35-39'
    elif 40 <= age <= 44:
        return '40-44'
    elif 45 <= age <= 49:
        return '45-49'
    elif 50 <= age <= 54:
        return '50-54'
    elif 55 <= age <= 59:
        return '55-59'
    elif 60 <= age <= 64:
        return '60-64'
    elif 65 <= age <= 69:
        return '65-69'
    elif 70 <= age <= 74:
        return '70-74'
    elif 75 <= age <= 79:
        return '75-79'
    elif age >= 80:
        return '80 or older'
    else:
        return 'Unknown' # Fallback

# --- Removed Database Connection Function ---

# --- Removed Database Interaction Functions (save_prediction_history, update_last_test_record) ---

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # --- Data Preprocessing ---
        # Create a DataFrame with the expected feature names
        # The order MUST match the order the model was trained on.
        # Adjust feature names based on the actual model requirements.
        # The model expects 'AgeCategory', not raw 'Age'.
        feature_names = [
            'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
            'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
            'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
            'SkinCancer'
            # Removed 'Age' as the model uses 'AgeCategory'
        ]
        # Create a mapping for Age to AgeCategory if needed by the model
        # Example: Map actual age to categories like '18-24', '25-29', etc.
        # This depends heavily on how the model was trained.
        # For now, assuming the model takes numerical age directly or it's handled
        # If AgeCategory is needed, you'll need to implement the mapping logic here.
        # Example placeholder if AgeCategory is needed:
        # age_map = { ... } # Define your age to category mapping
        # data['AgeCategory'] = age_map.get(int(data.get('Age', 0)), default_category)

        # Ensure all expected features are present, provide defaults if necessary
        # Derive AgeCategory from Age
        age = data.get('age')
        if age is None:
            return jsonify({'error': 'Missing input feature: Age'}), 400
        try:
            age_category = get_age_category(int(age))
            if age_category == 'Unknown':
                 return jsonify({'error': f'Invalid Age provided: {age}'}), 400
        except ValueError:
             return jsonify({'error': f'Invalid Age format: {age}'}), 400

        # Prepare input data dictionary, using derived AgeCategory
        input_data = {}
        for feature in feature_names:
            if feature == 'AgeCategory':
                input_data[feature] = age_category
            else:
                # Use .get() with a default (e.g., 0 or specific default based on feature type)
                # Ensure the default '0' is appropriate for all features if missing.
                # Consider more specific defaults if needed.
                input_data[feature] = data.get(feature, 0)

        # Add a check to ensure all required features are now in input_data
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
             # This case should ideally not happen with the .get() defaults, but good practice
             return jsonify({'error': f'Internal error: Missing features after processing: {missing_features}'}), 500

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure correct data types (example, adjust as needed)
        # df = df.astype({'Age': int, 'BMI': float, ...})

        # --- Prediction ---
        prediction_proba = model.predict_proba(df)[0] # Get probabilities for class 0 and 1
        prediction = np.argmax(prediction_proba) # Get the class with the highest probability
        confidence = prediction_proba[prediction] # Get the confidence score for the predicted class

        # --- Removed Conditional Database Saving Logic ---

        # Return prediction result and confidence
        return jsonify({
            'prediction': int(prediction), # Ensure prediction is JSON serializable (int)
            'confidence': float(confidence) # Ensure confidence is JSON serializable (float)
        })

    except KeyError as e:
        # Handle cases where expected keys are missing in the input JSON
        return jsonify({'error': f'Missing input feature: {e}'}), 400
    except ValueError as e:
        # Handle cases where data cannot be converted to the expected type (e.g., float, int)
        return jsonify({'error': f'Invalid data format: {e}'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}") # Log the error server-side
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    
    app.run(debug=True) # Set debug=False for production
