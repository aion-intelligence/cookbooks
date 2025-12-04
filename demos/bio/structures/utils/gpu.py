import torch
import logging
import os

logger = logging.getLogger(__name__)

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = True

def check_gpu():
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        return True
    else:
        logger.warning("⚠️  No GPU detected - prediction will be slow")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return False