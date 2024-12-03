import torch.nn as nn


class ExtendedDecoder(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        self.original_decoder = original_decoder
        self.original_decoder.conv_out = nn.Identity()
        self.additional_layers = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.original_decoder(x)
        x = self.additional_layers(x)
        return x
