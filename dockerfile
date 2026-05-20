# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN apt-get update && apt-get install -y libgomp1 \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/src/serving/model/003f949ba8c3439b9fc053f162f98ba8/artifacts/model

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["sh", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port $PORT"]