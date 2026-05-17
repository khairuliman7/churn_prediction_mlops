import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.models import infer_signature
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

"""
This pipeline has below process in order:
    Load data
    Preprocess data
    Feature engineering data
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data                    
from src.data.preprocess import preprocess_data            
from src.features.build_features import build_features       

"""
This main training pipeline orchestrate the entire ML workflow
"""

def main(args):
    
    #MLflow use local file-based tracking
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  

    #mlflow started logging
    with mlflow.start_run():
        
        #log hyperparameter and config
        mlflow.log_param("model", "lightgbm")           
        mlflow.log_param("threshold", args.threshold)  
        mlflow.log_param("test_size", args.test_size) 

        #load
        print("Loading data...")
        df = load_data(args.input)  
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        #skip validation part since the code have problem, will deal with this later
        mlflow.log_metric("data_is_loaded", 1)
        print("Data validation skipped.")

        #preprocess
        print("Preprocessing data...")
        df = preprocess_data(df)  

        #save the processed dataset
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        #feature engineering
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc = build_features(df, target_col=target)  
        
        #convert boolean to integer
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        #save feature metadata for consistency
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        #get features
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        #save the feature locally for CI CD
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        #MLflow logging for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        #save artifacts for serving pipeline, ensure training and serving have consistent transformation
        preprocessing_artifact = {
            "feature_columns": feature_cols,  
            "target": target                  
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        #train test split
        print("Splitting data...")
        X = df_enc.drop(columns=[target]) 
        y = df_enc[target]                
        
        #stratify to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    
            stratify=y,                  
            random_state=42              
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        #handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        #model training
        print("Training LightGBM model...")
        
        #this models are trained with optimized hyperparameters
        model = LGBMClassifier(
            n_estimators=442,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=59,              
            min_child_samples=49, 
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1                 
        )

        #train model and track the training time
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        mlflow.log_metric("train_time", train_time)  
        print(f"Model trained in {train_time:.2f} seconds")

        #Model evaluation
        print("Evaluating model performance...")
        
        #generate prediction and track the time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  
        
        #treshold is 0.35
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  

        #log the essential metrics for model comparison and monitoring
        precision = precision_score(y_test, y_pred)   
        recall = recall_score(y_test, y_pred)         
        f1 = f1_score(y_test, y_pred)                 
        roc_auc = roc_auc_score(y_test, proba)        
        
        #log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Model Performance:")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        #model serialization and logging
        print("Saving model to MLflow...")
        
        #log model in standard format for serving
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model, 
            artifact_path="model", 
            signature = signature,
            input_example=X_train.iloc[:5],
            pip_requirements=[
                "lightgbm",
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )
        print("Model saved to MLflow for serving pipeline")

        #final summary
        print(f"\nPerformance Summary:")
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {pred_time:.4f}s")
        print(f"Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with LightGBM + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
Use this below to run the pipeline:

python scripts/main_pipeline.py --input data/customer_churn_1M.csv --target churn
"""