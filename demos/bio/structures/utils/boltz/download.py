import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_boltz_models(
    models_dir: Path,
    force_download: bool = False,
    revision: str = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
) -> None:
    """Download Boltz-2 model weights from HuggingFace"""
    if not force_download and check_models_exist(models_dir):
        print(f"Models already downloaded at {models_dir}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Boltz-2 models to {models_dir}")
    logger.info(f"Revision: {revision}")

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=revision,
        local_dir=str(models_dir),
        force_download=force_download,
    )


def check_models_exist(models_dir: Path) -> bool:
    """Check if Boltz-2 models are already downloaded"""
    if not models_dir.exists():
        return False

    required_files = ["config.json", "model.safetensors"]
    return all((models_dir / f).exists() for f in required_files)