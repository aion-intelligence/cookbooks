"""
Health check utilities for BioNeMo NIMs.
"""

import asyncio
import httpx
from typing import Dict, Optional
import logging

from .nim_client import DEFAULT_ENDPOINTS, NIMEndpoint

logger = logging.getLogger(__name__)


async def check_nim_health(
    endpoint: NIMEndpoint,
    timeout: float = 10.0,
) -> Dict[str, any]:
    """
    Check health of a single NIM endpoint.

    Args:
        endpoint: NIM endpoint configuration
        timeout: Request timeout in seconds

    Returns:
        Dictionary with health status and details.
    """
    result = {
        "name": endpoint.name,
        "url": endpoint.base_url,
        "healthy": False,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(
                f"{endpoint.base_url}{endpoint.health_endpoint}"
            )
            result["healthy"] = response.status_code == 200
            result["status_code"] = response.status_code
    except httpx.ConnectError:
        result["error"] = "Connection refused - NIM may not be running"
    except httpx.TimeoutException:
        result["error"] = "Health check timed out"
    except Exception as e:
        result["error"] = str(e)

    return result


async def check_all_nims(
    endpoints: Optional[Dict[str, NIMEndpoint]] = None,
    timeout: float = 10.0,
) -> Dict[str, Dict]:
    """
    Check health of all configured NIM endpoints.

    Args:
        endpoints: Dictionary of NIM endpoints (uses defaults if None)
        timeout: Request timeout per endpoint

    Returns:
        Dictionary mapping NIM names to health status.
    """
    endpoints = endpoints or DEFAULT_ENDPOINTS

    tasks = [check_nim_health(ep, timeout) for ep in endpoints.values()]
    results = await asyncio.gather(*tasks)

    return {r["name"]: r for r in results}


def check_all_nims_sync(
    endpoints: Optional[Dict[str, NIMEndpoint]] = None,
    timeout: float = 10.0,
) -> Dict[str, Dict]:
    """
    Synchronous version of check_all_nims.

    Args:
        endpoints: Dictionary of NIM endpoints
        timeout: Request timeout per endpoint

    Returns:
        Dictionary mapping NIM names to health status.
    """
    return asyncio.run(check_all_nims(endpoints, timeout))


def print_health_status(status: Dict[str, Dict]) -> None:
    """
    Print formatted health status.

    Args:
        status: Health status dictionary from check_all_nims
    """
    print("\nBioNeMo NIM Health Status")
    print("=" * 40)

    all_healthy = True
    for name, info in status.items():
        if info["healthy"]:
            icon = "\u2705"  # Green check
            state = "HEALTHY"
        else:
            icon = "\u274c"  # Red X
            state = "UNHEALTHY"
            all_healthy = False

        print(f"{icon} {name:12} {state:10} {info['url']}")
        if info.get("error"):
            print(f"   Error: {info['error']}")

    print("=" * 40)
    overall = "ALL HEALTHY" if all_healthy else "DEGRADED"
    healthy_count = sum(1 for v in status.values() if v["healthy"])
    print(f"Overall: {overall} ({healthy_count}/{len(status)} healthy)")
