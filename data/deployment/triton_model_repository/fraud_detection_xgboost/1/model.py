import json
import numpy as np
import xgboost as xgb
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        self.model_dir = args['model_repository']
        model_path = f"{self.model_dir}/1/model.json"

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load feature names 
        feature_path = f"{self.model_dir}/1/feature_names.txt"
        with open(feature_path, 'r') as f:
            self.feature_names = f.read().strip().split('\n')

        print(f"Model loaded with {len(self.feature_names)} features")

    def execute(self, requests):
        """Execute inference on batch of requests."""
        responses = []

        for request in requests:
            # Get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_features")
            input_data = input_tensor.as_numpy()

            # Create DMatrix WITH feature names 
            dmatrix = xgb.DMatrix(input_data, feature_names=self.feature_names)

            # Predict
            predictions = self.model.predict(dmatrix)

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "fraud_probability",
                predictions.astype(np.float32).reshape(-1, 1)
            )

            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        pass
