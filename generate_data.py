import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y

# Save to CSV
df.to_csv('data.csv', index=False)
print("Dataset created: data.csv")
print(f"Shape: {df.shape}")
print(f"Class distribution: {df['target'].value_counts().to_dict()}")
