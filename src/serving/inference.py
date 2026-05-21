import pandas as pd
from src.utils.logger import logger
from typing import Optional

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

from src.serving.preprocessing import preprocess
from src.serving.model_loader import load_model

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

def predict(input_dict: dict, request_id: Optional[str] = None) -> str:
    
    #convert input to dataframe
    df = pd.DataFrame([input_dict])
    
    #feature transformation
    logger.info(f"[{request_id}] Starting preprocessing")
    df_enc = preprocess(df)
    
    # Match MLflow expected schema exactly
    int64_cols = [
        "age", "dependents", "tenure", "paperless_billing",
        "senior_citizen", "num_services", "has_phone_service",
        "has_internet_service", "has_online_security",
        "has_online_backup", "has_device_protection",
        "has_tech_support", "has_streaming_tv",
        "has_streaming_movies", "num_service_calls",
        "late_payments", "days_since_last_interaction"
    ]

    float64_cols = [
        "annual_income", "monthlycharges", "totalcharges",
        "customer_satisfaction", "num_complaints",
        "avg_monthly_gb", "credit_score"
    ]

    int32_cols = [
        "gender_Male", "gender_Other",
        "education_college", "education_high_school",
        "education_master", "education_phd",
        "marital_status_married", "marital_status_single",
        "marital_status_widowed",
        "contract_one_year", "contract_two_year",
        "payment_method_credit_card",
        "payment_method_electronic_check",
        "payment_method_mailed_check"
    ]

    for col in int64_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype("int64")

    for col in float64_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype("float64")

    for col in int32_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype("int32")

    #generate prediction
    try:
        logger.info("Loading model for inference")

        logger.info(f"[{request_id}] Loading model")
        model = load_model()

        logger.info(f"Input shape: {df_enc.shape}")
        
        logger.info(f"[{request_id}] Running prediction")
        preds = model.predict(df_enc)

        logger.info(f"Prediction generated: {preds}")
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        logger.exception("Prediction pipeline failed")

        raise Exception(f"Model prediction failed: {e}")
    
    #convert binary prediction output to business language
    if result == 1:
        return "Likely to churn"     
    else:
        return "Not likely to churn" 
    
    logger.info(f"[{request_id}] Prediction completed: {result}")