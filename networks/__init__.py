import logging
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class SimpleImageAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # Input shape expected: [Batch, Channels, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),         # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),        # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),       # 8x8 -> 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.Sigmoid(),  # Assumes input pixel values are normalized between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SimpleImageDiscirminator(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.discriminator = nn.Sequential(
            # Input shape: [Batch, Channels, 64, 64]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),         # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(), # Flattens the 256 channels of 8x8 features into a vector
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.discriminator(x)