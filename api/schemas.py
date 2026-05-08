from pydantic import BaseModel
from typing import List, Optional, Dict


class PredictRequest(BaseModel):
    features: List[float]


class BatchPredictRequest(BaseModel):
    transaction_indices: List[int]


class PredictionResult(BaseModel):
    fraud_probability: float
    risk_level: str       # HIGH | MEDIUM | LOW | NONE
    is_fraud: bool


class PredictResponse(BaseModel):
    transaction_index: Optional[int] = None
    result: PredictionResult
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_transactions: int
    fraud_count: int
    high_risk_count: int
    avg_latency_ms: float
    throughput_tps: float


class TransactionInfo(BaseModel):
    index: int
    transaction_id: Optional[int] = None
    features: Dict[str, float]        # all 466 features (only when fetching single tx)
    top_features: Dict[str, float]    # top 10 most important features


class HealthResponse(BaseModel):
    api_status: str
    triton_status: str
    model_status: str
    total_test_transactions: int
    data_loaded: bool
