from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    triton_host: str = "triton"
    triton_http_port: int = 8000
    model_name: str = "fraud_detection_xgboost"
    features_path: str = "/data/features/test_features.parquet"
    feature_cols_path: str = "/data/features/feature_cols.txt"

    model_config = {"env_prefix": "FRAUD_"}


settings = Settings()
