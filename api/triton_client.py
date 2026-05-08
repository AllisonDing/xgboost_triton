"""
TritonInferenceClient wraps the tritonclient HTTP library.

Call flow for a single prediction:
  1. Build an InferInput tensor named "input_features"  (FP32, shape [B, 466])
  2. Call client.infer() — Triton routes to the Python backend model
  3. The Python backend runs xgb.predict_proba() and returns "fraud_probability"
  4. Output shape is [B, 1]; we flatten to a 1-D array of probabilities
"""

import tritonclient.http as httpclient
import numpy as np

from .config import settings


class TritonInferenceClient:
    def __init__(self):
        url = f"{settings.triton_host}:{settings.triton_http_port}"
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = settings.model_name

    # ── Health checks ─────────────────────────────────────────────────────────

    def is_server_healthy(self) -> bool:
        return self.client.is_server_live() and self.client.is_server_ready()

    def is_model_ready(self) -> bool:
        return self.client.is_model_ready(self.model_name)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: FP32 array of shape [batch_size, num_features]
        Returns:
            Fraud probabilities, shape [batch_size], values in [0, 1]
        """
        batch_size, num_features = features.shape

        infer_input = httpclient.InferInput(
            name="input_features",
            shape=[batch_size, num_features],
            datatype="FP32",
        )
        infer_input.set_data_from_numpy(features)

        infer_output = httpclient.InferRequestedOutput("fraud_probability")

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
        )

        # Shape: [batch_size, 1] — flatten to [batch_size]
        return response.as_numpy("fraud_probability").flatten()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_model_metadata(self) -> dict:
        return self.client.get_model_metadata(self.model_name)

    def get_server_statistics(self) -> dict:
        return self.client.get_server_statistics()
