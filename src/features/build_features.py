import pandas as pd

"""
this function will be used in the function below
"""

def _map_binary_series(s: pd.Series) -> pd.Series:
    
    #get unique values and remove missing values
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    #change Yes No values into 1, 0
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    
    #for any binary feature, use ordering to map to 1, 0
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    #for nonbinary, it will return unchanged
    return s

"""
this function is the third function in the pipeline

the preprocessed data would be passed here

the goal of this function is to make the features ready for machine learning:
    Binary Encoding
    One hot Encoding
    Feature Engineering
    Fill missing values
"""

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    #identify feature types (categoric or numeric)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    #split binary columns and multi columns from categorical columns
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-category: {multi_cols}")

    #apply binary encoding using previous function
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"{c}: {original_dtype} → binary (0/1)")

    #convert also the boolean features (if there are)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    #use one hot encoding, with drop_first = True to prevent multicollinearity
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns")

    #convert numerical to standard integer and fill missing values with 0 again
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df