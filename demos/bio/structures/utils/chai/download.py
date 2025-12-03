import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

def download_chai_models(models_dir: Path, revision: str = "main") -> None:
    """
    Download Chai-1 model weights from Hugging Face.
    
    Args:
        models_dir: Directory to store downloaded models
        revision: Git revision/branch to download (default: main)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Chai-1 models to {models_dir}")
    logger.info(f"Revision: {revision}")
    
    snapshot_download(
        repo_id="chaidiscovery/chai-1",
        local_dir=str(models_dir),
        revision=revision,
        local_dir_use_symlinks=False
    )
    
    logger.info("Model download complete")