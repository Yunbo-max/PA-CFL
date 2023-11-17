import torch
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA (GPU) is available!")
    # Your GPU-related code here
else:
    device = torch.device("cpu")
    print("CUDA (GPU) is not available. Falling back to CPU.")
    # Fallback to CPU-related code
