import os
import pandas as pd
import mlflow

"""
This pipeline is for ML Model production to serve with Consistense Feature
    Core Inference functionality for the prediction model
    Ensure serving feature transformation math training transformation

Key responsibility:
    Load ML flow logged model and feature metadata from training
    Apply identical feature transformations as used during training
    Ensure correct feature ordering for model input 
    Convert model predictions to user-friendly output
"""

MODEL_DIR = "/app/model"

"""
Model loading configuration

In production, uses model copied to container at build time
"""

#load the model in MLflow pyfunc format, ensure compatibility
try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    
    #if cannot, try loading from local ML flow tracking
    try:
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

#load the exact feature column order used during training
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

#mappings must exactly math those used in training
BINARY_MAP = {
    "paperless_billing": {"No": 0, "Yes": 1},     
}

NUMERIC_COLS = ["gender", "education", "marital_status, contract, payment_method"]

"""
This function 
    Ensures that the features are transformed exactly as they were during training
    Prevent train/serve skew

Transformation Pipeline
    Clean column name and handle data types
    Binary encoding
    One hot encoding
    Convert boolean to integers
    Align features with training schema and order
"""

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df.columns = df.columns.str.strip()
    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)
    
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                 
                .str.strip()                 
                .map(mapping)                
                .astype("Int64")             
                .fillna(0)                    
                .astype(int)                  
            )
    
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

"""
This is the main prediction function for customer churn inference

The function provides complete inference pipeline from raw customer data to prediction output

Called by both FastAPI and Gradio to ensure consistent prediction

Pipeline:
    Convert input to dataframe
    Apply feature transformation
    Generate model prediction using LightGBM model
    Convert prediction to user-friendly string
"""

def predict(input_dict: dict) -> str:
    
    #convert input to dataframe
    df = pd.DataFrame([input_dict])
    
    #feature transformation
    df_enc = _serve_transform(df)
    
    #generate prediction
    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    #convert binary prediction output to business language
    if result == 1:
        return "Likely to churn"     
    else:
        return "Not likely to churn" 