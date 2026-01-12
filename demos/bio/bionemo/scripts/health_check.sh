#!/bin/bash
# Health check for BioNeMo NIM containers
# Usage: ./health_check.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_nim() {
    local name=$1
    local port=$2

    printf "%-12s: " "$name"

    if ! docker ps -q -f name="$name-nim" | grep -q .; then
        echo -e "${YELLOW}NOT RUNNING${NC}"
        return 1
    fi

    if curl -s "http://localhost:$port/v1/health/ready" > /dev/null 2>&1; then
        echo -e "${GREEN}HEALTHY${NC}"
        return 0
    else
        echo -e "${RED}UNHEALTHY${NC}"
        return 1
    fi
}

echo "BioNeMo NIM Health Status"
echo "========================="
echo ""

esm2_ok=0
evo2_ok=0
geneformer_ok=0

check_nim "ESM-2" 8001 && esm2_ok=1
check_nim "Evo2" 8002 && evo2_ok=1
check_nim "Geneformer" 8003 && geneformer_ok=1

echo ""

total=$((esm2_ok + evo2_ok + geneformer_ok))
if [ $total -eq 3 ]; then
    echo -e "Overall: ${GREEN}ALL HEALTHY${NC}"
elif [ $total -eq 0 ]; then
    echo -e "Overall: ${RED}ALL DOWN${NC}"
else
    echo -e "Overall: ${YELLOW}DEGRADED ($total/3 healthy)${NC}"
fi
