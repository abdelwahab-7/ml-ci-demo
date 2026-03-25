import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Create synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log to MLflow
mlflow.set_experiment("model_validation_pipeline")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Save run ID to file for deployment job
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Training complete. Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run.info.run_id}")
    print("model_info.txt created")
