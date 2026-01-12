#!/bin/bash
# Stop BioNeMo NIM containers
# Usage: ./stop_nims.sh [all|esm2|evo2|geneformer]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

stop_container() {
    local name=$1
    if docker ps -q -f name="$name" | grep -q .; then
        log_info "Stopping $name..."
        docker stop "$name"
        docker rm "$name"
        log_info "$name stopped"
    else
        log_info "$name is not running"
    fi
}

# Main
TARGET=${1:-all}

case $TARGET in
    all)
        log_info "Stopping all BioNeMo NIMs..."
        stop_container "esm2-nim"
        stop_container "evo2-nim"
        stop_container "geneformer-nim"
        log_info "All NIMs stopped"
        ;;
    esm2)
        stop_container "esm2-nim"
        ;;
    evo2)
        stop_container "evo2-nim"
        ;;
    geneformer)
        stop_container "geneformer-nim"
        ;;
    *)
        echo "Usage: $0 [all|esm2|evo2|geneformer]"
        exit 1
        ;;
esac
