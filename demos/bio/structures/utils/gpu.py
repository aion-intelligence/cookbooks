import torch
import logging

logger = logging.getLogger(__name__)

def check_gpu():
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  No GPU detected - prediction will be slow")