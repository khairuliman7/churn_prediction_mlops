from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

#Inference logic behind the machine learning
from src.serving.inference import predict 

# Initialize FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

#Health check endpoint for AWS and load balancer
@app.get("/")
def root():
    return {"status": "ok"}

"""
This is the Pydantic model for automatic validation and API documentation

It is also the schema for requested customer data for the churn prediction

This schema defines all features required for the prediction, is should match for consistency
"""
class CustomerData(BaseModel):
    age: int                   # Age of customer as of prediction
    gender: str                # "Male", "Female", "Other"
    annual_income: float       # Annual income of customers
    education: str             # "college", "master", "high_school", "bachelor", "phd"
    marital_status: str        # "married", "widowed", "single", "divorced"
    dependents: int            # Amount of dependents of customers
    tenure: int                # Number of months subscribed with company
    contract: str              # "month_to_month", "one_year", "two_year"
    payment_method: str        # "electronic_check", "bank_transfer", "credit_card", "mailed_check"
    paperless_billing: str     # "Yes" or "No"
    senior_citizen: str        # "Yes" or "No"
    monthlycharges: float      # Monthly charges in dollars
    totalcharges: float        # Total charges to date
    num_services: int          # Number of service associated with the customer
    has_phone_service: str     # "Yes" or "No"
    has_internet_service: str  # "Yes" or "No"
    has_online_security: str   # "Yes" or "No"
    has_online_backup: str     # "Yes" or "No"
    has_device_protection: str # "Yes" or "No"
    has_tech_support: str      # "Yes" or "No"
    has_streaming_tv: str      # "Yes" or "No"
    has_streaming_movies: str  # "Yes" or "No"
    customer_satisfaction: float # Customer satisfaction rating from survey
    num_complaints: float      # Number of complaints made by this customer
    num_service_calls: int     # Number of service calls made by this customer
    late_payments: int         # Number of late payments made by the customer
    avg_monthly_gb: float      # Average GB of internet used monthly by customer
    days_since_last_interaction: int # How long since the customer interact with the company
    credit_score: float        # CTOS score of the customer

"""
This is the main prediction endpoint for churn prediction

This endpoint:
    Receive validated customer data via the Pydantic model
    Use the inference pipeline made to transform features and predict
    Return the churn prediction

The output should be either:
    Prediction: Likely to churn
    Prediction: Unlikely to churn
    Error: Prediction fails
"""
@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}