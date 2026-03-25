import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Intentionally create a BAD model with random labels
np.random.seed(123)  # Different seed for randomness
n_samples = 200
n_features = 5

# Generate random features
X = np.random.randn(n_samples, n_features)

# Use completely random labels instead of meaningful ones
y = np.random.randint(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log to MLflow
mlflow.set_experiment("model_validation_pipeline_bad")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression_BAD")
    mlflow.log_param("random_seed", 123)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Save run ID to file for deployment job
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Training complete. Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run.info.run_id}")
    print("❌ This model is intentionally BAD (accuracy < 0.85)")
    print("model_info.txt created")
