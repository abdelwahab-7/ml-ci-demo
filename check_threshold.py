import mlflow
import sys
import os

def check_model_accuracy(run_id=None, threshold=0.85):
    """Check if model accuracy meets threshold"""
    try:
        # Read run ID from file if not provided directly
        if run_id is None:
            with open("model_info.txt", "r") as f:
                run_id = f.read().strip()

        # Connect to MLflow (URI should be set via environment variable)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Get the run and its metrics
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")

        if accuracy is None:
            print(f"Error: Accuracy metric not found in run {run_id}")
            sys.exit(1)

        print(f"Run ID: {run_id}")
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Threshold: {threshold:.4f}")

        if accuracy >= threshold:
            print("✓ Model meets performance criteria!")
            return True
        else:
            print("✗ Model does NOT meet performance criteria!")
            print(f"Accuracy {accuracy:.4f} is below threshold {threshold:.4f}")
            sys.exit(1)

    except Exception as e:
        print(f"Error checking model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    check_model_accuracy(run_id, threshold)
