import random
import numpy as np
import torch

# -------------------------
# Replicability Logic
# -------------------------

def set_seed(seed=42):
    """
    Sets universal random seeds to ensure deterministic results 
    across Python, NumPy, and PyTorch.
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic algorithms to ensure GPU calculates the exact same gradients every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random Seed Set to: {seed}")

# -------------------------
# Hardware Diagnostics
# -------------------------

def get_device_capability():
    """
    Diagnostics to ensure the GPU is ready for LLM Fine-Tuning.
    Enables TF32 for newer NVIDIA architectures (L4).
    """
    # Enable TF32 for modern Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Go to Runtime > Change runtime type > Select NVIDIA L4.")

    device = torch.device('cuda')
    
    # Extract GPU Metadata
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    capability = torch.cuda.get_device_capability(0)
    bf16_support = torch.cuda.is_bf16_supported()

    # Print Status Report
    print(f"GPU Detected: {gpu_name}")
    print(f"    |-- Memory: {gpu_mem:.2f} GB")
    print(f"    |-- Compute Capability: {capability}")
    print(f"    |-- BFloat16 Support: {'Yes' if bf16_support else 'No'}")

    # Professional Warning System
    if "L4" not in gpu_name and "A100" not in gpu_name:
        print(f"\nWarning: Using {gpu_name}. Performance may be suboptimal for Gemma fine-tuning.")

    return device