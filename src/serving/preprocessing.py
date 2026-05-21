import os
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Feature schema path
FEATURE_FILE = os.getenv(
    "FEATURE_FILE",
    os.path.join(
        BASE_DIR,
        "model/003f949ba8c3439b9fc053f162f98ba8/artifacts/feature_columns.txt"
    )
)

# Load feature columns
try:
    with open(FEATURE_FILE, "r") as f:
        FEATURE_COLS = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(FEATURE_COLS)} feature columns")

except Exception as e:
    raise Exception(
        f"Failed to load feature columns from {FEATURE_FILE}: {e}"
    )

# Binary mappings
BINARY_MAP = {
    "paperless_billing": {"No": 0, "Yes": 1},
}

# Columns to numeric cast
NUMERIC_COLS = [
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method"
]

# MLflow schema dtype alignment
INT64_COLS = [
    "age", "dependents", "tenure", "paperless_billing",
    "senior_citizen", "num_services", "has_phone_service",
    "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv",
    "has_streaming_movies", "num_service_calls",
    "late_payments", "days_since_last_interaction"
]

FLOAT64_COLS = [
    "annual_income", "monthlycharges", "totalcharges",
    "customer_satisfaction", "num_complaints",
    "avg_monthly_gb", "credit_score"
]

INT32_COLS = [
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Clean columns
    df.columns = df.columns.str.strip()

    # Numeric conversion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    # Binary mapping
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

    # One-hot encoding
    obj_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
    ]

    if obj_cols:
        df = pd.get_dummies(
            df,
            columns=obj_cols,
            drop_first=True
        )

    # Boolean conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns

    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align with training schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    # Type alignment
    for col in INT64_COLS:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    for col in FLOAT64_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    for col in INT32_COLS:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    return df