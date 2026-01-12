"""
Unified API Gateway for BioNeMo models.

Provides a single entry point for all BioNeMo NIMs with:
- Unified routing to ESM-2, Evo2, and Geneformer
- Health check aggregation
- Request metadata and timing
"""

import ray
from ray import serve
from typing import Dict, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
import time
import logging

from .esm2_deployment import ESM2Deployment
from .evo2_deployment import Evo2Deployment
from .geneformer_deployment import GeneformerDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 20,
    },
)
class BioNeMoGateway:
    """
    Unified API Gateway for BioNeMo models.

    Routes requests to the appropriate model deployment:
    - /health - Health status of all models
    - /v1/biology/esm2/* - Protein embeddings and structure
    - /v1/biology/evo2/* - DNA sequence generation
    - /v1/biology/geneformer/* - Single-cell analysis
    """

    def __init__(
        self,
        esm2_handle: serve.DeploymentHandle,
        evo2_handle: serve.DeploymentHandle,
        geneformer_handle: serve.DeploymentHandle,
    ):
        """
        Initialize gateway with model handles.

        Args:
            esm2_handle: Handle to ESM-2 deployment
            evo2_handle: Handle to Evo2 deployment
            geneformer_handle: Handle to Geneformer deployment
        """
        self.models = {
            "esm2": esm2_handle,
            "evo2": evo2_handle,
            "geneformer": geneformer_handle,
        }
        logger.info("BioNeMoGateway initialized with all model handles")

    async def check_health(self) -> Dict:
        """
        Check health of all backend NIMs.

        Returns:
            Dictionary with health status per model.
        """
        status = {}
        for name, handle in self.models.items():
            try:
                healthy = await handle.check_health.remote()
                status[name] = "healthy" if healthy else "unhealthy"
            except Exception as e:
                status[name] = f"error: {str(e)}"

        return status

    async def health(self, request: Request) -> JSONResponse:
        """GET /health - Health status endpoint."""
        status = await self.check_health()

        all_healthy = all(s == "healthy" for s in status.values())
        healthy_count = sum(1 for s in status.values() if s == "healthy")

        return JSONResponse({
            "status": "healthy" if all_healthy else "degraded",
            "models": status,
            "healthy_count": f"{healthy_count}/{len(status)}",
            "timestamp": int(time.time()),
        })

    async def esm2_embeddings(self, request: Request) -> JSONResponse:
        """POST /v1/biology/esm2/embeddings - Protein embeddings."""
        body = await request.json()
        body["action"] = "embeddings"

        result = await self.models["esm2"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "esm2")

    async def esm2_structure(self, request: Request) -> JSONResponse:
        """POST /v1/biology/esm2/structure - Contact prediction."""
        body = await request.json()
        body["action"] = "contacts"

        result = await self.models["esm2"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "esm2")

    async def evo2_generate(self, request: Request) -> JSONResponse:
        """POST /v1/biology/evo2/generate - DNA generation."""
        body = await request.json()
        body["action"] = "generate"

        result = await self.models["evo2"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "evo2")

    async def evo2_embeddings(self, request: Request) -> JSONResponse:
        """POST /v1/biology/evo2/embeddings - DNA embeddings."""
        body = await request.json()
        body["action"] = "embeddings"

        result = await self.models["evo2"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "evo2")

    async def geneformer_embeddings(self, request: Request) -> JSONResponse:
        """POST /v1/biology/geneformer/embeddings - Cell embeddings."""
        body = await request.json()
        body["action"] = "embeddings"

        result = await self.models["geneformer"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "geneformer")

    async def geneformer_classify(self, request: Request) -> JSONResponse:
        """POST /v1/biology/geneformer/classify - Cell classification."""
        body = await request.json()
        body["action"] = "classify"

        result = await self.models["geneformer"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "geneformer")

    async def geneformer_perturb(self, request: Request) -> JSONResponse:
        """POST /v1/biology/geneformer/perturb - Perturbation analysis."""
        body = await request.json()
        body["action"] = "perturb"

        result = await self.models["geneformer"].remote(
            self._create_mock_request(body)
        )
        return self._format_response(result, "geneformer")

    def _create_mock_request(self, body: Dict) -> Request:
        """Create a mock request object for internal routing."""
        # This is a workaround - in production you'd pass the body directly
        from starlette.testclient import TestClient
        from starlette.applications import Starlette

        class MockRequest:
            async def json(self):
                return body

        return MockRequest()

    def _format_response(self, result: JSONResponse, model: str) -> JSONResponse:
        """Format response with standard metadata."""
        # If result is already a JSONResponse, extract the body
        if isinstance(result, JSONResponse):
            return result

        return JSONResponse({
            "id": f"bio-{int(time.time())}",
            "object": "biology.result",
            "created": int(time.time()),
            "model": model,
            "data": result,
        })

    async def __call__(self, request: Request) -> JSONResponse:
        """
        Route requests based on path.

        This is a fallback for direct deployment without ingress.
        """
        path = request.url.path

        if path == "/health" or path == "/":
            return await self.health(request)
        elif "/esm2/embeddings" in path:
            return await self.esm2_embeddings(request)
        elif "/esm2/structure" in path:
            return await self.esm2_structure(request)
        elif "/evo2/generate" in path:
            return await self.evo2_generate(request)
        elif "/evo2/embeddings" in path:
            return await self.evo2_embeddings(request)
        elif "/geneformer/embeddings" in path:
            return await self.geneformer_embeddings(request)
        elif "/geneformer/classify" in path:
            return await self.geneformer_classify(request)
        elif "/geneformer/perturb" in path:
            return await self.geneformer_perturb(request)
        else:
            return JSONResponse(
                {"error": f"Unknown endpoint: {path}"},
                status_code=404,
            )


def deploy_gateway(
    esm2_host: str = "localhost",
    esm2_port: int = 8001,
    evo2_host: str = "localhost",
    evo2_port: int = 8002,
    geneformer_host: str = "localhost",
    geneformer_port: int = 8003,
) -> serve.DeploymentHandle:
    """
    Deploy the BioNeMo Gateway with all model backends.

    Args:
        esm2_host: ESM-2 NIM host
        esm2_port: ESM-2 NIM port
        evo2_host: Evo2 NIM host
        evo2_port: Evo2 NIM port
        geneformer_host: Geneformer NIM host
        geneformer_port: Geneformer NIM port

    Returns:
        Deployment handle for the gateway.
    """
    # Create model deployments
    esm2 = ESM2Deployment.bind(nim_host=esm2_host, nim_port=esm2_port)
    evo2 = Evo2Deployment.bind(nim_host=evo2_host, nim_port=evo2_port)
    geneformer = GeneformerDeployment.bind(
        nim_host=geneformer_host, nim_port=geneformer_port
    )

    # Create gateway with model handles
    gateway = BioNeMoGateway.bind(
        esm2_handle=esm2,
        evo2_handle=evo2,
        geneformer_handle=geneformer,
    )

    # Deploy
    return serve.run(gateway, name="bionemo-gateway", route_prefix="/")


def deploy_single_model(
    model: str,
    nim_host: str = "localhost",
    nim_port: Optional[int] = None,
) -> serve.DeploymentHandle:
    """
    Deploy a single BioNeMo model.

    Useful for development when you don't have all NIMs running.

    Args:
        model: Model name ("esm2", "evo2", or "geneformer")
        nim_host: NIM host
        nim_port: NIM port (uses default if not specified)

    Returns:
        Deployment handle for the model.
    """
    default_ports = {"esm2": 8001, "evo2": 8002, "geneformer": 8003}

    if model not in default_ports:
        raise ValueError(f"Unknown model: {model}. Must be one of {list(default_ports.keys())}")

    port = nim_port or default_ports[model]

    if model == "esm2":
        from .esm2_deployment import deploy_esm2
        return deploy_esm2(nim_host, port)
    elif model == "evo2":
        from .evo2_deployment import deploy_evo2
        return deploy_evo2(nim_host, port)
    else:
        from .geneformer_deployment import deploy_geneformer
        return deploy_geneformer(nim_host, port)
