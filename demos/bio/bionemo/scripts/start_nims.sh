#!/bin/bash
# Start BioNeMo NIM containers
# Usage: ./start_nims.sh [all|esm2|evo2|geneformer]

set -e

# Configuration
NGC_API_KEY=${NGC_API_KEY:?"NGC_API_KEY environment variable must be set"}
LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-~/.cache/nim}

# Create cache directory
mkdir -p "$LOCAL_NIM_CACHE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

wait_for_health() {
    local name=$1
    local port=$2
    local max_attempts=${3:-60}
    local attempt=1

    log_info "Waiting for $name to be ready (port $port)..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/v1/health/ready" > /dev/null 2>&1; then
            log_info "$name is ready!"
            return 0
        fi
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done

    log_error "$name failed to start after $max_attempts attempts"
    return 1
}

start_esm2() {
    log_info "Starting ESM-2 NIM container..."

    # Stop existing container if running
    docker rm -f esm2-nim 2>/dev/null || true

    docker run -d --name esm2-nim --runtime=nvidia \
        --gpus '"device=0"' \
        -e NGC_API_KEY="$NGC_API_KEY" \
        -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
        -p 8001:8000 \
        nvcr.io/nim/nvidia/esm2:latest

    wait_for_health "ESM-2" 8001
}

start_evo2() {
    log_info "Starting Evo2 NIM container (requires 2 GPUs)..."

    # Stop existing container if running
    docker rm -f evo2-nim 2>/dev/null || true

    docker run -d --name evo2-nim --runtime=nvidia \
        --gpus '"device=1,2"' \
        -e NGC_API_KEY="$NGC_API_KEY" \
        -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
        -p 8002:8000 \
        nvcr.io/nim/arc/evo2:2

    wait_for_health "Evo2" 8002 120  # Longer timeout for large model
}

start_geneformer() {
    log_info "Starting Geneformer NIM container..."

    # Stop existing container if running
    docker rm -f geneformer-nim 2>/dev/null || true

    docker run -d --name geneformer-nim --runtime=nvidia \
        --gpus '"device=3"' \
        -e NGC_API_KEY="$NGC_API_KEY" \
        -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
        -p 8003:8000 \
        nvcr.io/nim/nvidia/geneformer:latest

    wait_for_health "Geneformer" 8003
}

# Main
TARGET=${1:-all}

case $TARGET in
    all)
        log_info "Starting all BioNeMo NIMs..."
        start_esm2
        start_evo2
        start_geneformer
        log_info "All NIMs started successfully!"
        ;;
    esm2)
        start_esm2
        ;;
    evo2)
        start_evo2
        ;;
    geneformer)
        start_geneformer
        ;;
    *)
        log_error "Unknown target: $TARGET"
        echo "Usage: $0 [all|esm2|evo2|geneformer]"
        exit 1
        ;;
esac

log_info "NIM endpoints:"
echo "  ESM-2:      http://localhost:8001"
echo "  Evo2:       http://localhost:8002"
echo "  Geneformer: http://localhost:8003"
