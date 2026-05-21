# 🚀 Customer Churn Prediction – End-to-End MLOps Project

> **Can we predict which customers are about to leave — before they actually do?**  
This project builds a complete machine learning system that not only predicts customer churn but also deploys the model as a production-ready API with Docker and CI/CD automation.

It demonstrates an end-to-end MLOps workflow: from data preprocessing and model training to deployment, monitoring, and continuous integration.

---

## 📌 Project Overview

Customer churn is one of the most critical problems in subscription-based businesses.  
The goal of this project is to:

- Predict whether a customer will churn (binary classification)
- Deploy the trained model as a scalable REST API
- Containerize the application using Docker
- Implement CI/CD for automated builds and deployment
- Add lightweight prediction tracking for monitoring usage

This project is designed as a **practical MLOps pipeline**, not just a standalone ML model.

---

## 🧠 Key Features

- End-to-end ML pipeline (data → training → inference)
- REST API using **FastAPI**
- Dockerized application for portability
- CI/CD pipeline using GitHub Actions
- Prediction tracking system (mini monitoring layer)
- Structured project architecture for scalability
- JSON-based inference API
- Model persistence (saved and loaded for inference)

---

## 🏗️ System Architecture

User Request
↓
FastAPI Endpoint (/predict)
↓
Preprocessing Pipeline
↓
Trained ML Model (loaded at startup)
↓
Prediction Output (Churn / Not Churn)
↓
Logging / Tracking System


---

## 📁 Project Structure

Customer Churn/
│
│
├── code
│ ├── Code.ipynb 
│
├──scripts
│ ├── main_pipeline.py # Pipeline
│
├── src/
│ ├── app/
│ │ └── main.py # FastAPI application
│ │
│ ├── model/
│ │ ├── train.py # Model training pipeline
│ │ ├── predict.py # Inference logic
│ │ └── artifacts/ # Saved model files
│ │
│ ├── utils/
│ │ └── preprocessing.py # Data preprocessing steps
│
├── .gitignore
├── Dockerfile # Container setup
├── requirements.txt # Dependencies
├── .github/workflows/
│ └── ci.yml # CI/CD pipeline
└── README.md


---

## ⚙️ Tech Stack

- Python 🐍
- FastAPI
- Scikit-learn / ML models
- Pandas, NumPy
- Docker
- GitHub Actions (CI/CD)
- Uvicorn (API server)

---

## 📊 Model Lifecycle
Data preprocessing and feature engineering
Model training using classification algorithms
Model evaluation and selection
Model serialization (saved artifact)
Loaded at API startup for inference
Monitoring and Logging
CI/CD pipeline