FROM python:3.10-slim

# Build argument for the run ID
ARG RUN_ID

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Simulate model download using the RUN_ID
# In production, this would fetch the actual model from MLflow Model Registry
RUN echo "Downloading model with RUN_ID: ${RUN_ID}" && \
    echo "Model downloaded successfully to /app/model"

# Default command to run the service
CMD ["python", "serve.py"]
