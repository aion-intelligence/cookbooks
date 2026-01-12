# BioNeMo NIMs Integration

Serve NVIDIA BioNeMo pre-trained biomolecular AI models via Ray Serve for production inference.

## Models

| Model | Domain | Capabilities | GPU Requirements |
|-------|--------|--------------|------------------|
| **ESM-2** | Proteins | Sequence embeddings, contact prediction | 1x A100-40GB |
| **Evo2** | Genomics | DNA generation, embeddings | 2x H100-80GB |
| **Geneformer** | Single-cell | Cell embeddings, classification | 1x A100-40GB |

## Quick Start

### 1. Prerequisites

```bash
# Get NGC API key from https://ngc.nvidia.com
export NGC_API_KEY="your-api-key"

# Install dependencies
pip install -r requirements.txt
```

### 2. Start NIM Containers

```bash
# Start all NIMs (requires 4+ GPUs)
./scripts/start_nims.sh all

# Or start individual models
./scripts/start_nims.sh esm2       # ESM-2 on port 8001
./scripts/start_nims.sh evo2       # Evo2 on port 8002
./scripts/start_nims.sh geneformer # Geneformer on port 8003
```

### 3. Check Health

```bash
./scripts/health_check.sh
```

### 4. Run Demo Notebook

```bash
jupyter notebook esm2_serving.ipynb
```

## API Endpoints

When running the unified gateway:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health status of all models |
| `/v1/biology/esm2/embeddings` | POST | Protein sequence embeddings |
| `/v1/biology/esm2/structure` | POST | Contact map prediction |
| `/v1/biology/evo2/generate` | POST | DNA sequence generation |
| `/v1/biology/evo2/embeddings` | POST | DNA embeddings |
| `/v1/biology/geneformer/embeddings` | POST | Cell embeddings |
| `/v1/biology/geneformer/classify` | POST | Cell type classification |

## Example Requests

### ESM-2: Protein Embeddings

```bash
curl -X POST http://localhost:8000/esm2 \
  -H "Content-Type: application/json" \
  -d '{
    "action": "embeddings",
    "sequences": ["GIVEQCCTSICSLYQLENYCN"]
  }'
```

### Evo2: DNA Generation

```bash
curl -X POST http://localhost:8000/evo2 \
  -H "Content-Type: application/json" \
  -d '{
    "action": "generate",
    "sequence": "ATCGATCGATCG",
    "num_tokens": 100
  }'
```

### Geneformer: Cell Embeddings

```bash
curl -X POST http://localhost:8000/geneformer \
  -H "Content-Type: application/json" \
  -d '{
    "action": "embeddings",
    "cells": [{"TP53": 100.5, "BRCA1": 50.2, "MYC": 200.0}]
  }'
```

## Python Usage

### Direct NIM Client

```python
from utils.nim_client import NIMClient, DEFAULT_ENDPOINTS
import asyncio

async def get_embeddings():
    async with NIMClient(DEFAULT_ENDPOINTS["esm2"]) as client:
        result = await client.predict({
            "sequences": ["GIVEQCCTSICSLYQLENYCN"],
            "include_embeddings": True,
        })
    return result

result = asyncio.run(get_embeddings())
```

### Ray Serve Deployment

```python
from ray import serve
from deployments import deploy_gateway

# Deploy all models
handle = deploy_gateway()

# Or deploy single model
from deployments import deploy_single_model
handle = deploy_single_model("esm2")
```

### Batch Inference with Ray Data

```python
import ray
from ray import data as ray_data
from utils.nim_client import SyncNIMClient, DEFAULT_ENDPOINTS

class ESM2Predictor:
    def __init__(self):
        self.client = SyncNIMClient(DEFAULT_ENDPOINTS["esm2"])

    def __call__(self, batch):
        result = self.client.predict({
            "sequences": batch["sequence"].tolist(),
            "include_embeddings": True,
        })
        return {"embedding": [r["embedding"] for r in result["results"]]}

# Process dataset
ds = ray_data.from_items([{"sequence": "MKTVRQ..."}, ...])
results = ds.map_batches(ESM2Predictor, batch_size=32)
```

## Architecture

```
┌─────────────────┐
│   Ray Serve     │  ← Unified API Gateway
│   Gateway       │
└────────┬────────┘
         │
    ┌────┴────┬──────────┐
    │         │          │
┌───▼───┐ ┌───▼───┐ ┌────▼────┐
│ESM-2  │ │ Evo2  │ │Geneformer│  ← Ray Serve Deployments
│Deploy │ │Deploy │ │ Deploy   │     (CPU-only proxies)
└───┬───┘ └───┬───┘ └────┬────┘
    │         │          │
┌───▼───┐ ┌───▼───┐ ┌────▼────┐
│ESM-2  │ │ Evo2  │ │Geneformer│  ← BioNeMo NIM Containers
│ NIM   │ │ NIM   │ │   NIM    │     (GPU inference)
│:8001  │ │:8002  │ │  :8003   │
└───────┘ └───────┘ └──────────┘
```

## Directory Structure

```
demos/bio/bionemo/
├── README.md
├── requirements.txt
├── configs/
│   ├── esm2_config.yaml
│   ├── evo2_config.yaml
│   ├── geneformer_config.yaml
│   └── gateway_config.yaml
├── utils/
│   ├── __init__.py
│   ├── nim_client.py        # Async/sync HTTP clients
│   └── health_checks.py     # Health monitoring
├── deployments/
│   ├── __init__.py
│   ├── esm2_deployment.py   # ESM-2 Ray Serve wrapper
│   ├── evo2_deployment.py   # Evo2 Ray Serve wrapper
│   ├── geneformer_deployment.py
│   └── gateway.py           # Unified API gateway
├── scripts/
│   ├── start_nims.sh        # Start NIM containers
│   ├── stop_nims.sh         # Stop NIM containers
│   └── health_check.sh      # Check NIM health
└── esm2_serving.ipynb       # Demo notebook
```

## Configuration

### Local Development

```python
from utils.cluster import init_ray, ClusterMode

init_ray(mode=ClusterMode.LOCAL)
```

### Anyscale Production

```python
init_ray(mode=ClusterMode.ANYSCALE)
```

## Hardware Requirements

| Model | Min GPU | Recommended | Notes |
|-------|---------|-------------|-------|
| ESM-2 (650M) | 16GB | A100-40GB | Smallest, fastest |
| ESM-2 (3B) | 40GB | A100-80GB | Better embeddings |
| Evo2 (40B) | 160GB | 2x H100-80GB | Large model |
| Geneformer | 16GB | A100-40GB | Efficient |

## Troubleshooting

### NIM Container Won't Start

```bash
# Check Docker logs
docker logs esm2-nim

# Verify NGC authentication
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
```

### Health Check Fails

```bash
# Check if container is running
docker ps | grep nim

# Test NIM directly
curl http://localhost:8001/v1/health/ready
```

### Out of Memory

- Reduce batch size in requests
- Use smaller model variant (ESM-2 650M instead of 3B)
- Ensure no other GPU processes running

## Next Steps

After completing the Quick Start:

### 1. Deploy to Production (Anyscale)

```python
from deployments import deploy_gateway

# Switch to Anyscale mode
init_ray(mode=ClusterMode.ANYSCALE)

# Deploy with autoscaling
handle = deploy_gateway()
```

### 2. Add Authentication

Edit `deployments/gateway.py` to add API key validation:

```python
async def __call__(self, request: Request):
    api_key = request.headers.get("X-API-Key")
    if not validate_api_key(api_key):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    # ... rest of handler
```

### 3. Enable Rate Limiting

Update `configs/gateway_config.yaml`:

```yaml
rate_limiting:
  enabled: true
  default_rpm: 60
  burst_rpm: 100
```

### 4. Set Up Monitoring

- Ray Dashboard: `http://localhost:8265` (local) or Anyscale console
- Add custom metrics with `ray.util.metrics`
- Integrate with Prometheus/Grafana for production

### 5. Scale Individual Models

Adjust autoscaling in deployment files:

```python
autoscaling_config={
    "min_replicas": 2,      # Always keep 2 running
    "max_replicas": 10,     # Scale up to 10
    "target_num_ongoing_requests_per_replica": 5,
}
```

## Resources

- [NVIDIA BioNeMo Documentation](https://docs.nvidia.com/bionemo-framework/latest/)
- [BioNeMo GitHub](https://github.com/NVIDIA/bionemo-framework)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [NGC Container Registry](https://ngc.nvidia.com)
