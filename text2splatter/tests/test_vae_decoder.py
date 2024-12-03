import torch
from copy import deepcopy
from diffusers import AutoencoderKL


device = "cuda:1"

vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="vae"
        )

decoder = deepcopy(vae.decoder)

# We need input/output of this shape:
# input.shape: torch.Size([4, 256, 16, 16])
# output.shape: torch.Size([4, 24, 128, 128])
# vae = vae.to(device)
# vae_latent = vae.encode(torch.randn(4, 3, 512, 512).to(device)).latent_dist.mean
# print("vae_latent.shape:", vae_latent.shape)
# print(decoder)
# exit()

# dummy_input = torch.randn(4, 256, 16, 16).to(device)
dummy_input = torch.randn(4, 4, 64, 64).to(device)
# print(decoder.conv_in.weight.shape)
# print(decoder.conv_out.weight.shape)

# decoder.conv_in = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# decoder.conv_out = torch.nn.Conv2d(
#     in_channels=128,  # Input channels from the previous layer
#     out_channels=24,  # Desired output channels
#     kernel_size=4,    # Matches the stride for downscaling
#     stride=4,         # Reduces spatial dimensions by 4x
#     padding=0         # No extra padding
# )

'''
Before conv_out the outputs are of shape: torch.Size([4, 128, 512, 512])

After conv_out the outputs are of shape: torch.Size([4, 24, 512, 512])
'''
# decoder.conv_out = torch.nn.Conv2d(128, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

# with torch.no_grad():
# #     # decoder.conv_in.weight.copy_(vae.decoder.conv_in.weight.repeat(1, 256 // 4, 1, 1))
#     decoder.conv_out.weight.copy_(vae.decoder.conv_out.weight.repeat(24 // 3, 1, 1, 1))


# decoder = decoder.to(device)
# decoder.requires_grad_(False)
# decoder.up_blocks.requires_grad_(True)

# print(decoder.conv_in.weight.shape)
# print(decoder.conv_out.weight.shape)
# # output = decoder(dummy_input)
# output = decoder(vae_latent)

# print(output.shape)

# We need output of this shape: torch.Size([4, 24, 128, 128])

from torch import nn

class ExtendedDecoder(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        self.original_decoder = original_decoder
        self.original_decoder.conv_out = nn.Identity()
        self.additional_layers = nn.Sequential(
            nn.Conv2d(128, 63, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(63),
            nn.SiLU(),
            nn.Conv2d(63, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.original_decoder(x)
        x = self.additional_layers(x)
        return x

# Create the extended decoder
extended_decoder = ExtendedDecoder(decoder)
print(extended_decoder)
del decoder
with torch.no_grad():
    extended_decoder.additional_layers[0].weight.copy_(vae.decoder.conv_out.weight.repeat(63 // 3, 1, 1, 1))

extended_decoder = extended_decoder.to(device)

output = extended_decoder(dummy_input)
# print(output.shape)