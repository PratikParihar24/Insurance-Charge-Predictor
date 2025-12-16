import joblib
import numpy as np
import pandas as pd
from pathlib import Path  # New Import

# Define the base directory (the one containing src/, models/, etc.)
# This gets the current file's parent directory (src/) and then its parent (the root folder).
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the absolute path to the model file
MODEL_PATH = BASE_DIR / "models" / "optimized_random_forest_regressor.pkl"

# Load the trained model using the absolute path
# The script will crash here if the file is truly not there.
try:
    MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure you have run the model-saving step in your notebook.")
    # Re-raise the error to stop execution
    raise e

# Define the columns the model was trained on (must be exact!)
# This ensures new data has the same structure as the training data.
FEATURE_COLS = [
    'age', 'bmi', 'children', 'Is_Obese', 'bmi_age_interaction',
    'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 
    'region_southwest'
]

def preprocess_and_predict(input_data: dict) -> float:
    """
    Takes raw customer data, preprocesses it, and predicts the annual charge.
    
    Args:
        input_data: A dictionary of a single customer's features.
        
    Returns:
        The predicted annual insurance charge in dollars.
    """
    
    # 1. Convert input dictionary to a DataFrame for processing
    df_raw = pd.DataFrame([input_data])
    
    # 2. Replicate Feature Engineering steps
    df_raw['Is_Obese'] = (df_raw['bmi'] > 30).astype(int)
    df_raw['bmi_age_interaction'] = df_raw['bmi'] * df_raw['age']
    
    # 3. Replicate One-Hot Encoding
    # Start by creating all 4 region columns + sex/smoker columns.
    df_encoded = pd.get_dummies(df_raw, columns=['sex', 'smoker', 'region'], drop_first=False, dtype=int)
    
    # Manually ensure all required columns are present (setting missing to 0)
    # This step is critical because a new customer might not have a feature present
    # (e.g., if the customer is 'female' and we only have 'sex_male', the 'sex_female' column
    # might be missing, which breaks the model structure).
    for col in FEATURE_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # 4. Select and order features exactly as the model expects
    X_new = df_encoded[FEATURE_COLS]
    
    # 5. Predict the log charge
    log_charge_pred = MODEL.predict(X_new)
    
    # 6. Inverse Transform to get the final dollar amount
    charge_pred = np.exp(log_charge_pred[0])
    
    return charge_pred

# --- Example Usage ---
if __name__ == '__main__':
    # Hypothetical New Customer: 30-year-old male smoker from the southeast with high BMI
    new_customer = {
        'age': 30,
        'sex': 'male',
        'bmi': 35.5,
        'children': 1,
        'smoker': 'yes',
        'region': 'southeast'
    }
    
    predicted_charge = preprocess_and_predict(new_customer)
    
    print("--- New Customer Prediction ---")
    print(f"Input Data: {new_customer}")
    print(f"Predicted Annual Charge: ${predicted_charge:,.2f}")