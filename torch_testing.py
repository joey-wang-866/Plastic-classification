import torch
import torchvision

print(f"torch: {torch.__version__}, torchvision: {torchvision.__version__}")
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0))