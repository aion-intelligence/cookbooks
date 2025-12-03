from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_results(output_dir: Path) -> Dict[str, List[Path]]:
    """Extract and organize Boltz-2 results with proper directory traversal"""
    results = {
        "structures": [],
        "confidence": [],
        "metadata": [],
        "other": []
    }

    print(f"\nSearching for results in: {output_dir}")

    for file_path in output_dir.rglob("*"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(output_dir)

        if file_path.suffix == ".cif":
            results["structures"].append(file_path)
            logger.debug(f"  Found structure: {rel_path}")
        elif file_path.suffix == ".json" and "confidence" in file_path.name.lower():
            results["confidence"].append(file_path)
            logger.debug(f"  Found confidence: {rel_path}")
        elif file_path.suffix == ".json":
            results["metadata"].append(file_path)
            logger.debug(f"  Found metadata: {rel_path}")
        elif file_path.name != "input.yaml":
            results["other"].append(file_path)
            logger.debug(f"  Found other: {rel_path}")

    if not results["structures"]:
        logger.warning("\n⚠️  No CIF files found - checking for alternative output formats")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                logger.warning(f"    {file_path.relative_to(output_dir)}")

    return results

