"""
Evo2 Ray Serve deployment wrapping BioNeMo NIM.

Evo2 is a large-scale DNA foundation model that provides:
- DNA sequence generation
- Sequence embeddings
- Zero-shot prediction
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

# Valid nucleotide alphabet
VALID_NUCLEOTIDES = set("ACGT")


def validate_dna_sequence(sequence: str) -> Optional[str]:
    """
    Validate DNA sequence.

    Returns error message if invalid, None if valid.
    """
    if not sequence:
        return "Empty sequence"

    invalid_chars = set(sequence.upper()) - VALID_NUCLEOTIDES
    if invalid_chars:
        return f"Invalid nucleotides: {invalid_chars}"

    if len(sequence) > 8192:
        return f"Sequence too long ({len(sequence)} > 8192)"

    return None


@serve.deployment(
    ray_actor_options={"num_cpus": 1},  # CPU only - GPU is in NIM container
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,  # Limited due to large GPU requirements
        "target_num_ongoing_requests_per_replica": 5,
    },
    health_check_period_s=60,
    health_check_timeout_s=30,
)
class Evo2Deployment:
    """
    Ray Serve deployment for Evo2 DNA foundation model.

    Routes requests to BioNeMo Evo2 NIM container.

    Note: Evo2 is a large model (40B parameters) requiring 2x H100 GPUs.

    Endpoints:
    - /generate: Generate DNA sequences
    - /embeddings: Get sequence embeddings
    """

    def __init__(
        self,
        nim_host: str = "localhost",
        nim_port: int = 8002,
        timeout: float = 300.0,  # Longer timeout for large model
    ):
        """
        Initialize Evo2 deployment.

        Args:
            nim_host: Host where Evo2 NIM is running
            nim_port: Port for Evo2 NIM
            timeout: Request timeout in seconds
        """
        self.nim_url = f"http://{nim_host}:{nim_port}"
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.nim_url,
            timeout=httpx.Timeout(timeout),
        )
        logger.info(f"Evo2Deployment initialized, NIM at {self.nim_url}")

    async def check_health(self) -> bool:
        """Health check for Ray Serve."""
        try:
            response = await self._client.get("/v1/health/ready")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Evo2 health check failed: {e}")
            return False

    async def generate(
        self,
        sequence: str,
        num_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Dict:
        """
        Generate DNA sequence continuation.

        Args:
            sequence: Starting DNA sequence (prompt)
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Dictionary with generated sequence.
        """
        error = validate_dna_sequence(sequence)
        if error:
            return {"error": error, "sequence": None}

        payload = {
            "sequence": sequence.upper(),
            "num_tokens": num_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        response = await self._client.post(
            "/v1/biology/arc/evo2/generate",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def get_embeddings(
        self,
        sequences: List[str],
    ) -> Dict:
        """
        Get DNA sequence embeddings.

        Args:
            sequences: List of DNA sequences

        Returns:
            Dictionary with embeddings for each sequence.
        """
        for i, seq in enumerate(sequences):
            error = validate_dna_sequence(seq)
            if error:
                return {
                    "error": f"Invalid sequence at index {i}: {error}",
                    "embeddings": [],
                }

        payload = {
            "sequences": [s.upper() for s in sequences],
            "include_embeddings": True,
        }

        response = await self._client.post(
            "/v1/biology/arc/evo2/embeddings",
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def __call__(self, request: Request) -> JSONResponse:
        """Handle HTTP requests."""
        body = await request.json()
        action = body.get("action", "generate")

        start_time = time.time()

        try:
            if action == "generate":
                result = await self.generate(
                    sequence=body.get("sequence", ""),
                    num_tokens=body.get("num_tokens", 100),
                    temperature=body.get("temperature", 1.0),
                    top_k=body.get("top_k", 50),
                    top_p=body.get("top_p", 0.95),
                )

            elif action == "embeddings":
                sequences = body.get("sequences", [])
                result = await self.get_embeddings(sequences)

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
            "model": "evo2",
            "latency_ms": (time.time() - start_time) * 1000,
        }

        return JSONResponse(result)


def deploy_evo2(
    nim_host: str = "localhost",
    nim_port: int = 8002,
) -> serve.DeploymentHandle:
    """
    Deploy Evo2 to Ray Serve.

    Args:
        nim_host: Evo2 NIM host
        nim_port: Evo2 NIM port

    Returns:
        Deployment handle for Evo2.
    """
    deployment = Evo2Deployment.bind(nim_host=nim_host, nim_port=nim_port)
    return serve.run(deployment, name="evo2", route_prefix="/evo2")
