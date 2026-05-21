import os
import pandas as pd
import mlflow

"""
This pipeline is part of serving in the Machine Learning model

Key responsibility:
    Load ML flow logged model and feature metadata from training
"""

# ALWAYS use ONE consistent model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(
        BASE_DIR,
        "model/003f949ba8c3439b9fc053f162f98ba8/artifacts/model"
    )
)

"""
Model loading configuration

In production, uses model copied to container at build time
"""

#load the model in MLflow pyfunc format, ensure compatibility
_model = None

def load_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_DIR)
    return _model