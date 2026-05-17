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

"""
Gradio interface function processes form inputs and return the prediction

This function:
    Takes individual form inputs from the UI
    Constructs the data dictionary matching the API schema
    Use the same inference pipeline used by the API
    Returns the prediction
"""
def gradio_interface(
    age, gender, annual_income, education, marital_status, dependents, tenure, 
    contract, payment_method, paperless_billing, senior_citizen, monthlycharges, 
    totalcharges, num_services, has_phone_service, has_internet_service, has_online_security, 
    has_online_backup, has_device_protection, has_tech_support, has_streaming_tv, 
    has_streaming_movies, customer_satisfaction, num_complaints, num_service_calls, 
    late_payments, avg_monthly_gb, days_since_last_interaction, credit_score
):
    data = {
        "age": int(age),
        "gender": gender,
        "annual_income": float(annual_income),
        "education": education,
        "marital_status": marital_status, 
        "dependents": int(dependents), 
        "tenure": int(tenure), 
        "contract": contract, 
        "payment_method": payment_method, 
        "paperless_billing": paperless_billing, 
        "senior_citizen": senior_citizen, 
        "monthlycharges": float(monthlycharges), 
        "totalcharges": float(totalcharges), 
        "num_services": int(num_services), 
        "has_phone_service": has_phone_service, 
        "has_internet_service": has_internet_service, 
        "has_online_security": has_online_security, 
        "has_online_backup": has_online_backup, 
        "has_device_protection": has_device_protection, 
        "has_tech_support": has_tech_support, 
        "has_streaming_tv": has_streaming_tv, 
        "has_streaming_movies": has_streaming_movies, 
        "customer_satisfaction": float(customer_satisfaction), 
        "num_complaints": float(num_complaints),
        "num_service_calls": int(num_service_calls), 
        "late_payments": int(late_payments), 
        "avg_monthly_gb": float(avg_monthly_gb), 
        "days_since_last_interaction": int(days_since_last_interaction), 
        "credit_score": float(credit_score)
    }
    
    #Call the predict function similarly used by API
    result = predict(data)
    return str(result)  

"""
Gradio Configuration

Comprehensive Gradio interface with all features
"""
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label = "Age", value = 40, minimum = 0, maximum = 100),
        gr.Dropdown(["Male", "Female", "Other"], label = "Gender", value = "Male"),
        gr.Number(label = "Annual Income", value=4000.0, minimum = 0, maximum = 100000),
        gr.Dropdown(["College", "Master", "High School", "Bachelor", "PHD"], label = "Education", value="High School"),
        gr.Dropdown(["Married", "Widowed", "Single", "Divorced"], label = "Marital Status", value="Single"),
        gr.Number(label="Number of Dependents", value=0, minimum=0, maximum=15),
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=120),
        gr.Dropdown(["Month to Month", "One Year", "Two Year"], label="Contract", value="Month to Month"),
        gr.Dropdown(["Electronic Check", "Bank Transfer", "Credit Card", "Mailed Check"], label="Payment Method", value="Bank Transfer"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billings", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Senior Citizen", value="No"),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=2000),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=100000),
        gr.Number(label="Number of Services", value=1, minimum=0, maximum=15),
        gr.Dropdown(["Yes", "No"], label="Phone Service (Yes or No)", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Internet Service (Yes or No)", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Online Security (Yes or No)", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Online Backup (Yes or No)", value="No"),
        gr.Dropdown(["Yes", "No"], label="Device Protection (Yes or No)", value="No"),
        gr.Dropdown(["Yes", "No"], label="Technical Support (Yes or No)", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Streaming TV (Yes or No)", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Streaming Movies (Yes or No)", value="Yes"),
        gr.Number(label="Customer Satisfaction 0-10", value=5.0, minimum=0, maximum=10),
        gr.Number(label="Number of Complaints", value=1.0, minimum=0, maximum=20),
        gr.Number(label="Number of Service Calls made", value=0, minimum=0, maximum=20),
        gr.Number(label="Number of Late Payments made", value=0, minimum=0, maximum=20),
        gr.Number(label="Average GB used (Monthly)", value=1.0, minimum=0, maximum=600),
        gr.Number(label="Days since Last Interaction", value=1, minimum=0, maximum=2000),
        gr.Number(label="Credit Score", value=600.0, minimum=350.0, maximum=800.0),
    ],
    outputs = gr.Textbox(label="Churn Prediction", lines=2),
    title = "Customer Churn Predictor",
    description = 
    """
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.
    """,
    examples=[
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", 
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 
         1, 85.0, 85.0],
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
    ],
    theme=gr.themes.Soft() 
)

app = gr.mount_gradio_app(
    app,           
    demo,          
    path="/ui"     
)