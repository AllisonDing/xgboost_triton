import tritonclient.http as httpclient
import numpy as np
import pandas as pd
import time

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")
model_name = "fraud_detection_xgboost"

# Load features and data
with open('data/features/feature_cols.txt', 'r') as f:
    features = f.read().strip().split('\n')

test_df = pd.read_parquet('data/features/test_features.parquet')

print(f"Connected to Triton | Model: {model_name} | Features: {len(features)}")
print(f"Loaded {len(test_df):,} test transactions\n")

# ============================================================================
# Single Prediction
# ============================================================================
print("="*60)
print("Single Transaction Prediction")
print("="*60)

# Prepare single transaction
X = test_df.iloc[[0]][features].fillna(-999).values.astype(np.float32)

# Create input/output
inputs = [httpclient.InferInput("input_features", X.shape, "FP32")]
inputs[0].set_data_from_numpy(X)
outputs = [httpclient.InferRequestedOutput("fraud_probability")]

# Predict
start = time.perf_counter()
response = client.infer(model_name, inputs, outputs=outputs)
prob = float(response.as_numpy("fraud_probability")[0][0])
latency = (time.perf_counter() - start) * 1000

print(f"Transaction ID: {test_df.iloc[0]['TransactionID']}")
print(f"Fraud Probability: {prob:.4f}")
print(f"Risk Level: {'HIGH' if prob > 0.9 else 'MEDIUM' if prob > 0.7 else 'LOW' if prob > 0.5 else 'NO'}")
print(f"Latency: {latency:.2f}ms\n")

# ============================================================================
# Bulk Prediction
# ============================================================================
print("="*60)
print("Transactions Prediction")
print("="*60)

for i in range(10):
    X = test_df.iloc[[i]][features].fillna(-999).values.astype(np.float32)
    
    inputs = [httpclient.InferInput("input_features", X.shape, "FP32")]
    inputs[0].set_data_from_numpy(X)
    outputs = [httpclient.InferRequestedOutput("fraud_probability")]

    start = time.perf_counter()
    response = client.infer(model_name, inputs, outputs=outputs)
    prob = float(response.as_numpy("fraud_probability")[0][0])
    latency = (time.perf_counter() - start) * 1000

    print(f"Transaction ID: {test_df.iloc[i]['TransactionID']}")
    print(f"Fraud Probability: {prob:.4f}")
    print(f"Risk Level: {'HIGH' if prob > 0.9 else 'MEDIUM' if prob > 0.7 else 'LOW' if prob > 0.5 else 'NO'}")
    print(f"Latency: {latency:.2f}ms\n")

# ============================================================================
# Batch Prediction
# ============================================================================
print("="*60)
print("Batch Performance Benchmark")
print("="*60)
print(f"\n{'Batch Size':<12} {'Time (s)':<12} {'Throughput (txn/s)':<20} {'Latency (ms)'}")
print("-"*60)

for batch_size in [1, 10, 100, 1000]:
    # Prepare batch
    X = test_df.head(batch_size)[features].fillna(-999).values.astype(np.float32)
    
    inputs = [httpclient.InferInput("input_features", X.shape, "FP32")]
    inputs[0].set_data_from_numpy(X)
    outputs = [httpclient.InferRequestedOutput("fraud_probability")]
    
    # Predict
    start = time.perf_counter()
    response = client.infer(model_name, inputs, outputs=outputs)
    elapsed = time.perf_counter() - start
    
    # Stats
    throughput = batch_size / elapsed
    latency = (elapsed / batch_size) * 1000
    
    print(f"{batch_size:<12} {elapsed:<12.4f} {throughput:<20.0f} {latency:.3f}")

print("\n" + "="*60)
print("Complete! Press Ctrl+C in Triton terminal to stop server.")
print("="*60)