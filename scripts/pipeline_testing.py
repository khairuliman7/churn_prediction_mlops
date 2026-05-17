import os
import sys
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

file_path = "/Users/khairul/Downloads/Machine Learning/Customer Churn/data/customer_churn_1M.csv"

df = load_data(file_path)
df = preprocess_data(df)
df = build_features(df)

assert df.shape[0] > 0
assert df.isnull().sum().sum() >= 0

print("Pipeline stage test PASSED")