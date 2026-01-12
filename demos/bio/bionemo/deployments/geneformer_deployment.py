"""
Geneformer Ray Serve deployment wrapping BioNeMo NIM.

Geneformer is a single-cell foundation model that provides:
- Cell embeddings from gene expression data
- Cell type classification
- In-silico perturbation analysis
"""

import ray
from ray import serve
from typing import Dict, List, Optional, Any
from starlette.requests import Request
from starlette.responses import JSONResponse
import httpx
import logging
import time

logger = logging.getLogger(__name__)


def validate_gene_expression(data: Dict[str, float]) -> Optional[str]:
    """
    Validate gene expression data.

    Returns error message if invalid, None if valid.
    """
    if not data:
        return "Empty gene expression data"

    if not isinstance(data, dict):
        return "Gene expression must be a dictionary"

    if len(data) > 2048:
        return f"Too many genes ({len(data)} > 2048)"

    for gene, value in data.items():
        if not isinstance(gene, str):
            return f"Gene name must be string, got {type(gene)}"
        if not isinstance(value, (int, float)):
            return f"Expression value must be numeric, got {type(value)}"

    return None


@serve.deployment(
    ray_actor_options={"num_cpus": 1},  # CPU only - GPU is in NIM container
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 10,
    },
    health_check_period_s=30,
    health_check_timeout_s=10,
)
class GeneformerDeployment:
    """
    Ray Serve deployment for Geneformer single-cell model.

    Routes requests to BioNeMo Geneformer NIM container.

    Endpoints:
    - /embeddings: Generate cell embeddings from gene expression
    - /classify: Classify cell types
    - /perturb: In-silico perturbation analysis
    """

    def __init__(
        self,
        nim_host: str = "localhost",
        nim_port: int = 8003,
        timeout: float = 120.0,
    ):
        """
        Initialize Geneformer deployment.

        Args:
            nim_host: Host where Geneformer NIM is running
            nim_port: Port for Geneformer NIM
            timeout: Request timeout in seconds
        """
        self.nim_url = f"http://{nim_host}:{nim_port}"
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.nim_url,
            timeout=httpx.Timeout(timeout),
        )
        logger.info(f"GeneformerDeployment initialized, NIM at {self.nim_url}")

    async def check_health(self) -> bool:
        """Health check for Ray Serve."""
        try:
            response = await self._client.get("/v1/health/ready")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Geneformer health check failed: {e}")
            return False

    async def get_embeddings(
        self,
        cells: List[Dict[str, float]],
    ) -> Dict:
        """
        Generate cell embeddings from gene expression data.

        Args:
            cells: List of gene expression dictionaries.
                   Each dict maps gene names to expression values.

        Returns:
            Dictionary with embeddings for each cell.
        """
        for i, cell in enumerate(cells):
            error = validate_gene_expression(cell)
            if error:
                return {
                    "error": f"Invalid cell at index {i}: {error}",
                    "embeddings": [],
                }

        payload = {
            "cells": cells,
            "include_embeddings": True,
        }

        response = await self._client.post(
            "/v1/biology/nvidia/geneformer",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def classify(
        self,
        cells: List[Dict[str, float]],
        cell_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Classify cell types from gene expression.

        Args:
            cells: List of gene expression dictionaries
            cell_types: Optional list of candidate cell types to classify into

        Returns:
            Dictionary with classification results.
        """
        for i, cell in enumerate(cells):
            error = validate_gene_expression(cell)
            if error:
                return {
                    "error": f"Invalid cell at index {i}: {error}",
                    "classifications": [],
                }

        payload = {
            "cells": cells,
            "task": "classify",
        }
        if cell_types:
            payload["cell_types"] = cell_types

        response = await self._client.post(
            "/v1/biology/nvidia/geneformer",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def perturb(
        self,
        cell: Dict[str, float],
        genes_to_perturb: List[str],
        perturbation_type: str = "knockout",
    ) -> Dict:
        """
        Perform in-silico perturbation analysis.

        Args:
            cell: Gene expression dictionary for a single cell
            genes_to_perturb: List of genes to perturb
            perturbation_type: Type of perturbation ("knockout", "overexpress")

        Returns:
            Dictionary with perturbation results.
        """
        error = validate_gene_expression(cell)
        if error:
            return {"error": error, "result": None}

        if not genes_to_perturb:
            return {"error": "No genes to perturb specified", "result": None}

        payload = {
            "cell": cell,
            "genes_to_perturb": genes_to_perturb,
            "perturbation_type": perturbation_type,
            "task": "perturb",
        }

        response = await self._client.post(
            "/v1/biology/nvidia/geneformer",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def __call__(self, request: Request) -> JSONResponse:
        """Handle HTTP requests."""
        body = await request.json()
        action = body.get("action", "embeddings")

        start_time = time.time()

        try:
            if action == "embeddings":
                cells = body.get("cells", [])
                result = await self.get_embeddings(cells)

            elif action == "classify":
                cells = body.get("cells", [])
                cell_types = body.get("cell_types")
                result = await self.classify(cells, cell_types)

            elif action == "perturb":
                cell = body.get("cell", {})
                genes = body.get("genes_to_perturb", [])
                perturb_type = body.get("perturbation_type", "knockout")
                result = await self.perturb(cell, genes, perturb_type)

            else:
                result = {"error": f"Unknown action: {action}"}

        except httpx.HTTPStatusError as e:
            result = {
                "error": f"NIM request failed: {e.response.status_code}",
                "detail": e.response.text,
            }
        except Exception as e:
            result = {"error": str(e)}

        result["_metadata"] = {
            "model": "geneformer",
            "latency_ms": (time.time() - start_time) * 1000,
        }

        return JSONResponse(result)


def deploy_geneformer(
    nim_host: str = "localhost",
    nim_port: int = 8003,
) -> serve.DeploymentHandle:
    """
    Deploy Geneformer to Ray Serve.

    Args:
        nim_host: Geneformer NIM host
        nim_port: Geneformer NIM port

    Returns:
        Deployment handle for Geneformer.
    """
    deployment = GeneformerDeployment.bind(nim_host=nim_host, nim_port=nim_port)
    return serve.run(deployment, name="geneformer", route_prefix="/geneformer")
