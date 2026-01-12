"""
BioNeMo NIM client utilities for communicating with NIM containers.

Provides async HTTP clients for ESM-2, Evo2, and Geneformer NIMs.
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class NIMEndpoint:
    """Configuration for a BioNeMo NIM endpoint."""

    name: str
    host: str
    port: int
    model_type: str  # "esm2", "evo2", "geneformer"
    health_endpoint: str = "/v1/health/ready"
    inference_endpoint: str = "/v1/biology"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# Default NIM endpoints configuration
DEFAULT_ENDPOINTS = {
    "esm2": NIMEndpoint(
        name="esm2",
        host="localhost",
        port=8001,
        model_type="esm2",
        inference_endpoint="/v1/biology/nvidia/esm2",
    ),
    "evo2": NIMEndpoint(
        name="evo2",
        host="localhost",
        port=8002,
        model_type="evo2",
        inference_endpoint="/v1/biology/arc/evo2/generate",
    ),
    "geneformer": NIMEndpoint(
        name="geneformer",
        host="localhost",
        port=8003,
        model_type="geneformer",
        inference_endpoint="/v1/biology/nvidia/geneformer",
    ),
}


class NIMClient:
    """
    Async client for BioNeMo NIM containers.

    Supports ESM-2, Evo2, and Geneformer NIMs with automatic
    endpoint configuration and health checking.

    Example:
        async with NIMClient(DEFAULT_ENDPOINTS["esm2"]) as client:
            if await client.health_check():
                result = await client.predict({"sequences": ["MKTVRQ..."]})
    """

    def __init__(
        self,
        endpoint: NIMEndpoint,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "NIMClient":
        self._client = httpx.AsyncClient(
            base_url=self.endpoint.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("NIMClient must be used as async context manager")
        return self._client

    async def health_check(self) -> bool:
        """
        Check if NIM container is ready.

        Returns:
            True if NIM is healthy, False otherwise.
        """
        try:
            response = await self.client.get(self.endpoint.health_endpoint)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {self.endpoint.name}: {e}")
            return False

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send inference request to NIM.

        Args:
            payload: Request payload (model-specific format)

        Returns:
            Response from NIM as dictionary.

        Raises:
            httpx.HTTPStatusError: If request fails.
        """
        response = await self.client.post(
            self.endpoint.inference_endpoint,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def predict_with_retry(
        self,
        payload: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send inference request with automatic retries.

        Args:
            payload: Request payload
            max_retries: Override default max retries

        Returns:
            Response from NIM.
        """
        retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                return await self.predict(payload)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    # Server error, retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} failed for {self.endpoint.name}: {e}"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    # Client error, don't retry
                    raise
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed for {self.endpoint.name}: {e}"
                )
                await asyncio.sleep(2**attempt)

        raise last_error


class SyncNIMClient:
    """
    Synchronous client for BioNeMo NIM containers.

    Use this for batch inference with Ray Data where async is not needed.

    Example:
        client = SyncNIMClient(DEFAULT_ENDPOINTS["esm2"])
        result = client.predict({"sequences": ["MKTVRQ..."]})
    """

    def __init__(
        self,
        endpoint: NIMEndpoint,
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=endpoint.base_url,
            timeout=httpx.Timeout(timeout),
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def health_check(self) -> bool:
        """Check if NIM container is ready."""
        try:
            response = self._client.get(self.endpoint.health_endpoint)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {self.endpoint.name}: {e}")
            return False

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send inference request to NIM."""
        response = self._client.post(
            self.endpoint.inference_endpoint,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


# Convenience functions for specific models


async def get_esm2_embeddings(
    sequences: List[str],
    host: str = "localhost",
    port: int = 8001,
    include_contacts: bool = False,
) -> Dict[str, Any]:
    """
    Get protein sequence embeddings from ESM-2 NIM.

    Args:
        sequences: List of amino acid sequences
        host: NIM host
        port: NIM port
        include_contacts: Include contact map predictions

    Returns:
        Dictionary with embeddings and optional contacts.
    """
    endpoint = NIMEndpoint(
        name="esm2",
        host=host,
        port=port,
        model_type="esm2",
        inference_endpoint="/v1/biology/nvidia/esm2",
    )

    async with NIMClient(endpoint) as client:
        return await client.predict(
            {
                "sequences": sequences,
                "include_embeddings": True,
                "include_contacts": include_contacts,
            }
        )


async def generate_dna_sequence(
    prompt: str,
    num_tokens: int = 100,
    host: str = "localhost",
    port: int = 8002,
    temperature: float = 1.0,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    Generate DNA sequence using Evo2 NIM.

    Args:
        prompt: Starting DNA sequence
        num_tokens: Number of tokens to generate
        host: NIM host
        port: NIM port
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        Dictionary with generated sequence.
    """
    endpoint = NIMEndpoint(
        name="evo2",
        host=host,
        port=port,
        model_type="evo2",
        inference_endpoint="/v1/biology/arc/evo2/generate",
    )

    async with NIMClient(endpoint) as client:
        return await client.predict(
            {
                "sequence": prompt,
                "num_tokens": num_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
        )


async def get_cell_embeddings(
    gene_expressions: List[Dict[str, float]],
    host: str = "localhost",
    port: int = 8003,
) -> Dict[str, Any]:
    """
    Get cell embeddings from Geneformer NIM.

    Args:
        gene_expressions: List of gene expression dictionaries
                         (gene_name -> expression_value)
        host: NIM host
        port: NIM port

    Returns:
        Dictionary with cell embeddings.
    """
    endpoint = NIMEndpoint(
        name="geneformer",
        host=host,
        port=port,
        model_type="geneformer",
        inference_endpoint="/v1/biology/nvidia/geneformer",
    )

    async with NIMClient(endpoint) as client:
        return await client.predict(
            {
                "gene_expressions": gene_expressions,
                "include_embeddings": True,
            }
        )
