import torch
from copy import deepcopy
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import AutoencoderKL

from text2splatter.models import SongUNetEncoder, load_encoder_weights
from text2splatter.splatter_image.datasets.dataset_factory import get_dataset

device = "cuda:1"

vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="vae"
        )

decoder = deepcopy(vae.decoder)
# print(decoder)

# We need input/output of this shape:
# input.shape: torch.Size([4, 256, 16, 16])
# output.shape: torch.Size([4, 24, 128, 128])

dummy_input = torch.randn(4, 256, 16, 16).to(device)
print(decoder.conv_in.weight.shape)
print(decoder.conv_out.weight.shape)

decoder.conv_in = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
decoder.conv_out = torch.nn.Conv2d(128, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

with torch.no_grad():
    decoder.conv_in.weight.copy_(vae.decoder.conv_in.weight.repeat(1, 256 // 4, 1, 1))
    decoder.conv_out.weight.copy_(vae.decoder.conv_out.weight.repeat(24 // 3, 1, 1, 1))


decoder = decoder.to(device)

print(decoder.conv_in.weight.shape)
print(decoder.conv_out.weight.shape)
output = decoder(dummy_input)
print(output.shape)