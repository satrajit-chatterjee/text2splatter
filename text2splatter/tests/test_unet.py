import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")
unet.conv_out = nn.Conv2d(320, 256, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
input_tensor = torch.randn(8, 320, 64, 64)  # Batch size = 8
output_tensor = unet.conv_out(input_tensor)

print(output_tensor.shape)