"""
ESM-2 Ray Serve deployment wrapping BioNeMo NIM.

ESM-2 is a protein language model that provides:
- Protein sequence embeddings (1280-dim for ESM2-650M)
- Per-residue predictions
- Contact map prediction for structure
"""

import ray
from ray import serve
from typing import Dict, List, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
import httpx
import logging
import time

logger = logging.getLogger(__name__)

# Valid amino acid alphabet
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_sequence(sequence: str) -> Optional[str]:
    """
    Validate protein sequence.

    Returns error message if invalid, None if valid.
    """
    if not sequence:
        return "Empty sequence"

    invalid_chars = set(sequence.upper()) - VALID_AMINO_ACIDS
    if invalid_chars:
        return f"Invalid amino acids: {invalid_chars}"

    if len(sequence) > 1024:
        return f"Sequence too long ({len(sequence)} > 1024)"

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
class ESM2Deployment:
    """
    Ray Serve deployment for ESM-2 protein language model.

    Routes requests to BioNeMo ESM-2 NIM container.

    Endpoints:
    - /embeddings: Generate protein sequence embeddings
    - /structure: Predict contact maps
    """

    def __init__(
        self,
        nim_host: str = "localhost",
        nim_port: int = 8001,
        timeout: float = 120.0,
    ):
        """
        Initialize ESM-2 deployment.

        Args:
            nim_host: Host where ESM-2 NIM is running
            nim_port: Port for ESM-2 NIM
            timeout: Request timeout in seconds
        """
        self.nim_url = f"http://{nim_host}:{nim_port}"
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.nim_url,
            timeout=httpx.Timeout(timeout),
        )
        logger.info(f"ESM2Deployment initialized, NIM at {self.nim_url}")

    async def check_health(self) -> bool:
        """Health check for Ray Serve."""
        try:
            response = await self._client.get("/v1/health/ready")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ESM-2 health check failed: {e}")
            return False

    async def get_embeddings(
        self,
        sequences: List[str],
        include_hiddens: bool = False,
    ) -> Dict:
        """
        Generate protein sequence embeddings.

        Args:
            sequences: List of amino acid sequences
            include_hiddens: Include hidden layer representations

        Returns:
            Dictionary with embeddings for each sequence.
        """
        # Validate sequences
        for i, seq in enumerate(sequences):
            error = validate_protein_sequence(seq)
            if error:
                return {
                    "error": f"Invalid sequence at index {i}: {error}",
                    "embeddings": [],
                }

        payload = {
            "sequences": [s.upper() for s in sequences],
            "include_embeddings": True,
            "include_hiddens": include_hiddens,
        }

        response = await self._client.post(
            "/v1/biology/nvidia/esm2",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def predict_contacts(
        self,
        sequence: str,
    ) -> Dict:
        """
        Predict contact map for protein structure.

        Args:
            sequence: Single amino acid sequence

        Returns:
            Dictionary with contact predictions.
        """
        error = validate_protein_sequence(sequence)
        if error:
            return {"error": error, "contacts": None}

        payload = {
            "sequences": [sequence.upper()],
            "include_contacts": True,
        }

        response = await self._client.post(
            "/v1/biology/nvidia/esm2",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def __call__(self, request: Request) -> JSONResponse:
        """
        Handle HTTP requests.

        Supports both direct calls and structured requests.
        """
        body = await request.json()
        action = body.get("action", "embeddings")

        start_time = time.time()

        try:
            if action == "embeddings":
                sequences = body.get("sequences", [])
                include_hiddens = body.get("include_hiddens", False)
                result = await self.get_embeddings(sequences, include_hiddens)

            elif action == "contacts" or action == "structure":
                sequence = body.get("sequence", "")
                result = await self.predict_contacts(sequence)

            else:
                result = {"error": f"Unknown action: {action}"}

        except httpx.HTTPStatusError as e:
            result = {
                "error": f"NIM request failed: {e.response.status_code}",
                "detail": e.response.text,
            }
        except Exception as e:
            result = {"error": str(e)}

        # Add metadata
        result["_metadata"] = {
            "model": "esm2",
            "latency_ms": (time.time() - start_time) * 1000,
        }

        return JSONResponse(result)


def deploy_esm2(
    nim_host: str = "localhost",
    nim_port: int = 8001,
) -> serve.DeploymentHandle:
    """
    Deploy ESM-2 to Ray Serve.

    Args:
        nim_host: ESM-2 NIM host
        nim_port: ESM-2 NIM port

    Returns:
        Deployment handle for ESM-2.
    """
    deployment = ESM2Deployment.bind(nim_host=nim_host, nim_port=nim_port)
    return serve.run(deployment, name="esm2", route_prefix="/esm2")
