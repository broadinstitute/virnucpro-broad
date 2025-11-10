"""GPU/device management utilities"""

import torch
import sys
import logging
from typing import Optional

logger = logging.getLogger('virnucpro.device')


def validate_and_get_device(device_str: str, fallback_to_cpu: bool = True) -> torch.device:
    """
    Validate device string and return torch.device object.

    Args:
        device_str: Device specification ("cpu", "cuda", "cuda:N", or "N")
        fallback_to_cpu: If True, fallback to CPU on errors (with warning)

    Returns:
        torch.device object

    Raises:
        ValueError: If device is invalid and fallback_to_cpu is False
    """
    # Handle auto-detection
    if device_str.lower() == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Auto-detected device: {device_str}")

    # Handle CPU
    if device_str.lower() == 'cpu':
        logger.info("Using CPU")
        return torch.device('cpu')

    # Check CUDA availability
    if not torch.cuda.is_available():
        error_msg = (
            "CUDA/GPU support is not available on this system.\n"
            "Possible causes:\n"
            "  - PyTorch was installed without CUDA support\n"
            "  - No NVIDIA GPU detected\n"
            "  - CUDA drivers not installed"
        )

        if fallback_to_cpu:
            logger.warning(f"{error_msg}\nFalling back to CPU")
            return torch.device('cpu')
        else:
            raise ValueError(f"{error_msg}\nPlease use --device cpu or install GPU support")

    # Parse device specification
    if device_str.lower() == 'cuda':
        device_id = 0
    elif device_str.startswith('cuda:'):
        try:
            device_id = int(device_str.split(':')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid device format: {device_str}")
    elif device_str.isdigit():
        device_id = int(device_str)
    else:
        raise ValueError(
            f"Invalid device specification: {device_str}\n"
            f"Valid formats: 'auto', 'cpu', 'cuda', 'cuda:N', or 'N'"
        )

    # Validate device ID is in range
    num_gpus = torch.cuda.device_count()
    if device_id >= num_gpus:
        error_msg = (
            f"GPU cuda:{device_id} is not available.\n"
            f"This system has {num_gpus} GPU(s): "
            f"{', '.join(f'cuda:{i}' for i in range(num_gpus))}\n"
            f"Use --list-devices to see available devices"
        )

        if fallback_to_cpu:
            logger.warning(f"{error_msg}\nFalling back to CPU")
            return torch.device('cpu')
        else:
            raise ValueError(error_msg)

    device = torch.device(f'cuda:{device_id}')

    # Log device info
    props = torch.cuda.get_device_properties(device_id)
    logger.info(f"Using device: {device}")
    logger.info(f"  GPU: {torch.cuda.get_device_name(device_id)}")
    logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")

    return device


def list_available_devices():
    """
    Print information about available compute devices.

    Used by the --list-devices CLI option.
    """
    print("Available compute devices:\n")

    # CPU (always available)
    print("  CPU:")
    print("    Device: cpu")
    print("    Available: Yes")

    # GPUs
    if torch.cuda.is_available():
        print(f"\n  GPUs ({torch.cuda.device_count()} detected):")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    Device: cuda:{i}")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")

            # Check if device is actually usable
            try:
                test = torch.zeros(1, device=f'cuda:{i}')
                del test
                print(f"      Status: Ready")
            except RuntimeError as e:
                print(f"      Status: Error - {e}")
    else:
        print("\n  GPUs:")
        print("    CUDA not available")
        print("    To enable GPU support:")
        print("      1. Install NVIDIA CUDA drivers")
        print("      2. Install PyTorch with CUDA support:")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu118")


def test_device(device: torch.device) -> bool:
    """
    Test if device is usable by allocating a small tensor.

    Args:
        device: Device to test

    Returns:
        True if device works, False otherwise
    """
    try:
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        return True
    except RuntimeError as e:
        logger.error(f"Device test failed for {device}: {e}")
        return False
