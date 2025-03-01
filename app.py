import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Get current working directory
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")  # Ensure this file exists

# Check if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Make sure it's included in your Render deployment.")

# Load the model
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    """Root route to check if the API is running."""
    return {"message": "FastAPI Model API is running!"}

# Define input schema
class UserInput(BaseModel):
    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: int

# Function to encode input data
def encode_user_input(user_input: UserInput):
    os_mapping = {i: f'OperatingSystems_{i}' for i in range(1, 9)}
    browser_mapping = {i: f'Browser_{i}' for i in range(1, 14)}
    region_mapping = {i: f'Region_{i}' for i in range(1, 10)}
    traffic_mapping = {i: f'TrafficType_{i}' for i in range(1, 21)}
    visitor_mapping = {
        'New_Visitor': 'VisitorType_New_Visitor',
        'Other': 'VisitorType_Other',
        'Returning_Visitor': 'VisitorType_Returning_Visitor'
    }

    # Feature names based on model training data
    feature_names = ['Administrative', 'Administrative_Duration', 'Informational',
                     'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                     'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Weekend',
                     'OperatingSystems_1', 'OperatingSystems_2', 'OperatingSystems_3',
                     'OperatingSystems_4', 'OperatingSystems_5', 'OperatingSystems_6',
                     'OperatingSystems_7', 'OperatingSystems_8', 'Browser_1', 'Browser_2',
                     'Browser_3', 'Browser_4', 'Browser_5', 'Browser_6', 'Browser_7',
                     'Browser_8', 'Browser_9', 'Browser_10', 'Browser_11', 'Browser_12',
                     'Browser_13', 'Region_1', 'Region_2', 'Region_3', 'Region_4',
                     'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9',
                     'TrafficType_1', 'TrafficType_2', 'TrafficType_3', 'TrafficType_4',
                     'TrafficType_5', 'TrafficType_6', 'TrafficType_7', 'TrafficType_8',
                     'TrafficType_9', 'TrafficType_10', 'TrafficType_11', 'TrafficType_12',
                     'TrafficType_13', 'TrafficType_14', 'TrafficType_15', 'TrafficType_16',
                     'TrafficType_17', 'TrafficType_18', 'TrafficType_19', 'TrafficType_20',
                     'VisitorType_New_Visitor', 'VisitorType_Other',
                     'VisitorType_Returning_Visitor']

    # Initialize all features with 0
    encoded_data = {feature: 0 for feature in feature_names}

    # Assign actual values
    for feature in ['Administrative', 'Administrative_Duration', 'Informational',
                    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Weekend']:
        encoded_data[feature] = getattr(user_input, feature)

    # One-hot encoding
    encoded_data[os_mapping.get(user_input.OperatingSystems, 'OperatingSystems_1')] = 1
    encoded_data[browser_mapping.get(user_input.Browser, 'Browser_1')] = 1
    encoded_data[region_mapping.get(user_input.Region, 'Region_1')] = 1
    encoded_data[traffic_mapping.get(user_input.TrafficType, 'TrafficType_1')] = 1
    encoded_data[visitor_mapping.get(user_input.VisitorType, 'VisitorType_New_Visitor')] = 1

    return pd.DataFrame([encoded_data])

# Function to load model and make predictions
def load_model_and_predict(user_input: UserInput):
    encoded_df = encode_user_input(user_input)
    
    try:
        prediction = model.predict(encoded_df)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Prediction endpoint
@app.post('/predict')
def predict(user_input: UserInput):
    try:
        prediction = load_model_and_predict(user_input)
        return {'prediction': prediction}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if '_name_' == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render dynamically assigns a port
    uvicorn.run(app, host="0.0.0.0", port=port)
