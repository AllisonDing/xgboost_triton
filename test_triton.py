import tritonclient.http as httpclient

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

print("="*60)
print("TRITON SERVER STATUS")
print("="*60)
print(f"Server Live: {client.is_server_live()}")
print(f"Server Ready: {client.is_server_ready()}")
print(f"Model Ready: {client.is_model_ready('fraud_detection_xgboost')}")
print("="*60)