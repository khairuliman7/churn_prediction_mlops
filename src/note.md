load_data.py

"""
Loads CSV data into a pandas DataFrame.

Args:
    file_path (str): Path to the CSV file.

Returns:
    pd.DataFrame: Loaded dataset.
"""

preprocess.py

"""
Basic cleaning for Telco churn.
- trim column names
- drop obvious ID cols
- fix TotalCharges to numeric
- map target Churn to 0/1 if needed
- simple NA handling
"""

build_features.py

"""
Apply deterministic binary encoding to 2-category features.

This function implements the core binary encoding logic that converts
categorical features with exactly 2 values into 0/1 integers. The mappings
are deterministic and must be consistent between training and serving.

"""

"""
Apply complete feature engineering pipeline for training data.

This is the main feature engineering function that transforms raw customer data
into ML-ready features. The transformations must be exactly replicated in the
serving pipeline to ensure prediction accuracy.

"""

validate_data.py

"""
Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

This function implements critical data quality checks that must pass before model training.
It validates data integrity, business logic constraints, and statistical properties
that the ML model expects.

"""

train.py

"""
Trains a LightGBM model and logs with MLflow.

Args:
    df (pd.DataFrame): Feature dataset.
    target_col (str): Name of the target column.
"""

tune.py

"""
Tunes an LightGBM model using Optuna.

Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Target.
"""

evaluate.py

"""
Evaluates an XGBoost model on test data.

Args:
    model: Trained model.
    X_test: Test features.
    y_test: Test labels.
"""

pipeline.py

"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

"""
Main training pipeline function that orchestrates the complete ML workflow.

"""

"""
Use this below to run the pipeline:

python scripts/main_pipeline.py --input data/customer_churn_1M.csv --target churn

"""

inference.py

"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

"""
Apply identical feature transformations as used during model training.

This function is CRITICAL for production ML - it ensures that features are
transformed exactly as they were during training to prevent train/serve skew.

Transformation Pipeline:
1. Clean column names and handle data types
2. Apply deterministic binary encoding (using BINARY_MAP)
3. One-hot encode remaining categorical features  
4. Convert boolean columns to integers
5. Align features with training schema and order

Args:
    df: Single-row DataFrame with raw customer data
    
Returns:
    DataFrame with features transformed and ordered for model input
    
IMPORTANT: Any changes to this function must be reflected in training
feature engineering to maintain consistency.
"""

"""
Main prediction function for customer churn inference.

This function provides the complete inference pipeline from raw customer data
to business-friendly prediction output. It's called by both the FastAPI endpoint
and the Gradio interface to ensure consistent predictions.

Pipeline:
1. Convert input dictionary to DataFrame
2. Apply feature transformations (identical to training)
3. Generate model prediction using loaded XGBoost model
4. Convert prediction to user-friendly string

Args:
    input_dict: Dictionary containing raw customer data with keys matching
                the CustomerData schema (18 features total)
                
Returns:
    Human-readable prediction string:
    - "Likely to churn" for high-risk customers (model prediction = 1)
    - "Not likely to churn" for low-risk customers (model prediction = 0)
    
Example:
    >>> customer_data = {
    ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
    ...     "MonthlyCharges": 85.0, ... # other features
    ... }
    >>> predict(customer_data)
    "Likely to churn"
"""

main.py

"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

