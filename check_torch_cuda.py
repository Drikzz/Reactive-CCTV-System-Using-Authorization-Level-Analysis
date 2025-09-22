# run to check if torch can access the GPU

# INSTALL PYTORCH FIRST:
# https://pytorch.org/get-started/locally/
# pick latest cuda version that matches your GPU and CUDA driver

# run this script to check gpu access
import torch
print(torch.cuda.is_available())   # True = GPU ready
print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None

