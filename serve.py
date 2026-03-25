#!/usr/bin/env python
import os

def main():
    run_id = os.getenv("RUN_ID", "unknown")
    print(f"Container started with RUN_ID: {run_id}")
    print("Model server is running...")
    # In production, this would load the model and start an API server
    # For demo, we just keep the container running
    import time
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
