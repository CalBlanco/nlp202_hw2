import torch

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Force initialization of MPS device
        torch.zeros(1).to(device)
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device
