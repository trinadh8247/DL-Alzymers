import torch
from src.logger import get_logger
from src.exception import AppException

logger = get_logger("utils")


def save_checkpoint(model, path):
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")
    except Exception as e:
        logger.exception("Failed to save checkpoint")
        raise AppException(f"Failed to save checkpoint to {path}", e)


def load_checkpoint(model, path, device):
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Loaded checkpoint from {path}")
        return model
    except Exception as e:
        logger.exception("Failed to load checkpoint")
        raise AppException(f"Failed to load checkpoint from {path}", e)
