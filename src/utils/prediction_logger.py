import os
import pandas as pd
from datetime import datetime

LOG_FILE = "logs/predictions.csv"

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)


def log_prediction(request_id: str, input_data: dict, prediction: str):
    """
    Save prediction to CSV for monitoring / analysis
    """

    row = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "input": str(input_data),
        "prediction": prediction
    }

    df = pd.DataFrame([row])

    file_exists = os.path.isfile(LOG_FILE)

    df.to_csv(
        LOG_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )