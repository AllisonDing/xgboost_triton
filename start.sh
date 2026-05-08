#!/usr/bin/env bash
# start.sh — quick-start script for the fraud detection demo
#
# Usage:
#   ./start.sh            # start all services
#   ./start.sh --down     # stop and remove containers
#   ./start.sh --logs     # tail all service logs

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${BLUE}[→]${NC} $*"; }

# ── Handle flags ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--down" ]]; then
  info "Stopping all services ..."
  docker compose down
  log "Done."
  exit 0
fi

if [[ "${1:-}" == "--logs" ]]; then
  docker compose logs -f
  exit 0
fi

# ── Check prerequisites ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "   Fraud Detection Demo  —  NVIDIA Triton + FastAPI"
echo "════════════════════════════════════════════════════════"
echo ""

# Docker
if ! command -v docker &>/dev/null; then
  err "Docker not found. Install Docker Desktop or Docker Engine first."
  exit 1
fi
log "Docker found: $(docker --version | head -1)"

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
  warn "nvidia-smi not found — Triton will run in CPU-only mode."
  warn "Edit docker-compose.yml to remove the 'deploy: resources: reservations' block."
else
  log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# Check that notebooks have been run
echo ""
info "Checking for required data files ..."

MISSING=0
for f in \
  "data/features/test_features.parquet" \
  "data/features/feature_cols.txt" \
  "data/deployment/triton_model_repository/fraud_detection_xgboost/config.pbtxt"; do
  if [[ -f "$f" ]]; then
    log "  $f"
  else
    err "  MISSING: $f"
    MISSING=1
  fi
done

if [[ $MISSING -eq 1 ]]; then
  echo ""
  warn "Some data files are missing. Run the training notebooks first:"
  warn "  1. jupyter notebook Part_6_supervised_learning.ipynb"
  warn "  2. jupyter notebook Part_7_ml_inference.ipynb"
  echo ""
  read -rp "Continue anyway? (containers will start but API will return 503) [y/N] " yn
  [[ "${yn,,}" != "y" ]] && exit 1
fi

# ── Start services ─────────────────────────────────────────────────────────────
echo ""
info "Building and starting services ..."
docker compose up --build -d

echo ""
info "Waiting for services to become healthy ..."
sleep 5

# Poll until all services are healthy (max 120 s)
DEADLINE=$(( $(date +%s) + 120 ))
while [[ $(date +%s) -lt $DEADLINE ]]; do
  HEALTHY=$(docker compose ps --format json 2>/dev/null \
    | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
total=0; ok=0
for l in lines:
    if not l: continue
    try:
        s = json.loads(l)
        total+=1
        if s.get('Health','') in ('healthy','') and s.get('State','')=='running':
            ok+=1
    except: pass
print(f'{ok}/{total}')
" 2>/dev/null || echo "?")

  if [[ "$HEALTHY" == "3/3" ]]; then
    break
  fi
  echo -n "."
  sleep 3
done
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
log " Frontend   →  http://localhost:3000"
log " API docs   →  http://localhost:3000/api/docs"
log " Triton     →  http://localhost:8000/v2/health/ready"
log " Metrics    →  http://localhost:8002/metrics"
echo "════════════════════════════════════════════════════════"
echo ""
info "Logs:  ./start.sh --logs"
info "Stop:  ./start.sh --down"
echo ""
