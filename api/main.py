"""
Fraud Detection API
===================
This service sits between the frontend and NVIDIA Triton Inference Server.

Architecture
------------
  Browser (port 3000)
      │  HTTP REST / JSON
      ▼
  FastAPI  (this service, port 8080)
      │  HTTP to Triton  (port 8000)
      ▼
  NVIDIA Triton Inference Server
      │  runs
      ▼
  XGBoost model  (Python backend, 466 features → fraud probability)

Endpoints
---------
  GET  /health                        System health (API + Triton + model)
  GET  /model/info                    Model metadata from Triton
  GET  /transactions?offset=&limit=   Paginated list of test transactions
  GET  /transactions/random           One random test transaction
  GET  /transactions/{index}          One test transaction by row index
  POST /predict                       Predict from a raw feature vector
  POST /predict/transaction/{index}   Predict a test transaction by index
  POST /predict/batch                 Batch predict (multiple indices in one Triton call)
"""

import logging
import random
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
    TransactionInfo,
)
from .triton_client import TritonInferenceClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# Top features by XGBoost importance (from Part_6 training notebook)
TOP_FEATURES = [
    "V258", "V91", "V201", "V70", "R_email_common",
    "V294", "V128", "V166", "V317", "V307",
]

# Module-level state populated during lifespan startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources once at startup; release on shutdown."""
    logger.info("Starting Fraud Detection API ...")

    _state["triton"] = TritonInferenceClient()
    logger.info("Triton client → %s:%d", settings.triton_host, settings.triton_http_port)

    # Test data is optional — the API starts even if notebooks haven't been run yet
    try:
        logger.info("Loading test features from %s", settings.features_path)
        df = pd.read_parquet(settings.features_path)
        with open(settings.feature_cols_path) as fh:
            feature_cols = [ln.strip() for ln in fh if ln.strip()]
        _state["df"] = df
        _state["feature_cols"] = feature_cols
        logger.info("Loaded %d transactions, %d features", len(df), len(feature_cols))
    except Exception as exc:
        logger.warning("Test data not available (%s). Run the training notebooks first.", exc)
        _state["df"] = None
        _state["feature_cols"] = []

    yield

    _state.clear()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Fraud Detection API",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.5:
        return "LOW"
    return "NONE"


def _require_data() -> None:
    if _state.get("df") is None:
        raise HTTPException(
            status_code=503,
            detail="Test data not loaded. Run Part_6_supervised_learning.ipynb first.",
        )


def _run_triton(features: np.ndarray) -> tuple[np.ndarray, float]:
    """Send features to Triton; return (probabilities, latency_ms)."""
    t0 = time.perf_counter()
    try:
        probs = _state["triton"].predict(features)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Triton inference failed: {exc}")
    return probs, (time.perf_counter() - t0) * 1000


def _build_tx_info(index: int, include_all: bool) -> TransactionInfo:
    df: pd.DataFrame = _state["df"]
    feature_cols: list = _state["feature_cols"]
    row = df.iloc[index]

    def safe(v):
        return float(v) if not pd.isna(v) else -999.0

    top = {f: safe(row[f]) for f in TOP_FEATURES if f in df.columns}
    tx_id = int(row["TransactionID"]) if "TransactionID" in df.columns else None
    all_feats = {f: safe(row[f]) for f in feature_cols} if include_all else {}

    return TransactionInfo(
        index=index,
        transaction_id=tx_id,
        features=all_feats,
        top_features=top,
    )


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check health of all system components."""
    triton: TritonInferenceClient = _state.get("triton")
    triton_ok = model_ok = False
    try:
        triton_ok = triton.is_server_healthy()
        model_ok = triton.is_model_ready()
    except Exception:
        pass

    df = _state.get("df")
    return HealthResponse(
        api_status="healthy",
        triton_status="healthy" if triton_ok else "unavailable",
        model_status="ready" if model_ok else "unavailable",
        total_test_transactions=len(df) if df is not None else 0,
        data_loaded=df is not None,
    )


@app.get("/model/info", tags=["System"])
async def model_info():
    """Retrieve model metadata from Triton (inputs, outputs, backend)."""
    try:
        meta = _state["triton"].get_model_metadata()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach Triton: {exc}")
    return {
        "model_name": meta["name"],
        "versions": meta.get("versions", []),
        "platform": meta.get("platform", ""),
        "inputs": meta.get("inputs", []),
        "outputs": meta.get("outputs", []),
        "num_features": len(_state.get("feature_cols", [])),
    }


# ── Transaction endpoints ─────────────────────────────────────────────────────

@app.get("/transactions", response_model=List[TransactionInfo], tags=["Transactions"])
async def list_transactions(
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    """Return a paginated slice of test transactions with top-feature values only."""
    _require_data()
    df = _state["df"]
    end = min(offset + limit, len(df))
    return [_build_tx_info(i, include_all=False) for i in range(offset, end)]


@app.get("/transactions/random", response_model=TransactionInfo, tags=["Transactions"])
async def random_transaction():
    """Return one randomly chosen test transaction (useful for demos)."""
    _require_data()
    idx = random.randint(0, len(_state["df"]) - 1)
    return _build_tx_info(idx, include_all=True)


@app.get("/transactions/{index}", response_model=TransactionInfo, tags=["Transactions"])
async def get_transaction(index: int):
    """Return a specific test transaction with all 466 features."""
    _require_data()
    n = len(_state["df"])
    if not (0 <= index < n):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range [0, {n - 1}]")
    return _build_tx_info(index, include_all=True)


# ── Prediction endpoints ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict fraud probability from a raw feature vector.
    The vector must contain exactly num_features float values
    in the same order as feature_cols.txt.
    """
    feature_cols = _state.get("feature_cols", [])
    if feature_cols and len(request.features) != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(request.features)}",
        )

    features = np.array(request.features, dtype=np.float32).reshape(1, -1)
    probs, latency_ms = _run_triton(features)
    prob = float(probs[0])

    return PredictResponse(
        result=PredictionResult(
            fraud_probability=prob,
            risk_level=_risk_level(prob),
            is_fraud=prob >= 0.5,
        ),
        latency_ms=latency_ms,
    )


@app.post("/predict/transaction/{index}", response_model=PredictResponse, tags=["Prediction"])
async def predict_transaction(index: int):
    """
    Predict fraud for a test transaction by row index.

    Step-by-step:
      1. Fetch the row from the pre-loaded test parquet
      2. Fill NaN → -999 (same preprocessing as training)
      3. Cast to FP32 and send to Triton via HTTP
      4. Triton routes to the XGBoost Python backend
      5. Return the fraud probability + risk level
    """
    _require_data()
    df: pd.DataFrame = _state["df"]
    feature_cols: list = _state["feature_cols"]

    n = len(df)
    if not (0 <= index < n):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range [0, {n - 1}]")

    row = df.iloc[index][feature_cols].fillna(-999)
    features = row.values.astype(np.float32).reshape(1, -1)

    probs, latency_ms = _run_triton(features)
    prob = float(probs[0])

    return PredictResponse(
        transaction_index=index,
        result=PredictionResult(
            fraud_probability=prob,
            risk_level=_risk_level(prob),
            is_fraud=prob >= 0.5,
        ),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict fraud for multiple test transactions in a single Triton call.

    Triton's dynamic batching merges concurrent requests and maximises GPU
    utilisation — this endpoint illustrates that by sending a whole batch at once.
    """
    _require_data()
    df: pd.DataFrame = _state["df"]
    feature_cols: list = _state["feature_cols"]
    indices = request.transaction_indices

    n = len(df)
    invalid = [i for i in indices if not (0 <= i < n)]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid indices (first 10): {invalid[:10]}")

    features = df.iloc[indices][feature_cols].fillna(-999).values.astype(np.float32)
    probs, total_ms = _run_triton(features)

    per_tx_ms = total_ms / len(indices)
    results = [
        PredictResponse(
            transaction_index=idx,
            result=PredictionResult(
                fraud_probability=float(p),
                risk_level=_risk_level(float(p)),
                is_fraud=float(p) >= 0.5,
            ),
            latency_ms=per_tx_ms,
        )
        for idx, p in zip(indices, probs)
    ]

    fraud_count = sum(1 for r in results if r.result.is_fraud)
    high_risk = sum(1 for r in results if r.result.risk_level == "HIGH")
    throughput = len(indices) / (total_ms / 1000)

    return BatchPredictResponse(
        results=results,
        total_transactions=len(indices),
        fraud_count=fraud_count,
        high_risk_count=high_risk,
        avg_latency_ms=per_tx_ms,
        throughput_tps=throughput,
    )
