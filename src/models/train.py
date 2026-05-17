import mlflow
import mlflow.lightgbm
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from mlflow.models import infer_signature


def train_model(df: pd.DataFrame, target_col: str):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1, 
        class_weight = "balanced"
    )

    with mlflow.start_run():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 6)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")