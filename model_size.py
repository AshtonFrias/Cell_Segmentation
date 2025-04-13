from models.SwinUnet import SwinUNet
from models.UNet import UNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = SwinUNet(1, device=device).to(device)
model = UNet(3, 1).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")