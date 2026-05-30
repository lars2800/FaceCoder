#=================================================================================================#
# based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py  #
# refactored the above code using Ai to work within the framework                                 #
#=================================================================================================#

from __future__ import annotations
import logging
import torch
import torch.nn as nn
from devices import DEVICE

LOGGER = logging.getLogger(__name__)

class AdvancedAdversarialNetwork:
    def __init__(self, latent_dim: int = 128) -> None:
        self.latent_dim = latent_dim

        self.model_auto_encoder = AdvancedImageAutoEncoder(in_channels=1, latent_dim=latent_dim)
        self.model_discriminator = AdvancedLatentDiscriminator(latent_dim=latent_dim)
    
    def to(self,device) -> AdvancedAdversarialNetwork:
        self.model_auto_encoder.to(device)
        self.model_discriminator.to(device)
        return self
    
    def save(self, filepath: str="checkpoints/model_advanced.pth") -> None:
        LOGGER.info(f"Saving model checkpoints to {filepath}...")
        checkpoint = {
            "autoencoder_state_dict": self.model_auto_encoder.state_dict(),
            "discriminator_state_dict": self.model_discriminator.state_dict(),
        }
        try:
            torch.save(checkpoint, filepath)
            LOGGER.info("Checkpoints saved successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to save checkpoints: {e}")
    
    def load(self, filepath: str = "checkpoints/model_advanced.pth") -> None:
        LOGGER.info(f"Loading model checkpoints from {filepath}...")
        try:
            checkpoint = torch.load(filepath, map_location=torch.device(DEVICE))
            self.model_auto_encoder.load_state_dict(checkpoint["autoencoder_state_dict"])
            self.model_discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            LOGGER.info("Checkpoints loaded successfully.")
        except FileNotFoundError:
            LOGGER.error(f"Checkpoint file not found at {filepath}")
        except Exception as e:
            LOGGER.error(f"Failed to load checkpoints: {e}")

class AdvancedImageAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128) -> None:
        super().__init__()
        
        # Output shape: [Batch, 256, 4, 4]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Flatten and project to latent space
        self.encoder_fc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Project back from latent space to start decoding
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder_conv = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def decode(self, z):
        z_projected = self.decoder_fc(z)
        z_reshaped = z_projected.view(z_projected.size(0), 256, 4, 4)
        return self.decoder_conv(z_reshaped)
    
    def forward(self, x):
        # Encode
        features = self.encoder_conv(x)
        features_flat = features.view(features.size(0), -1)
        latent_z = self.encoder_fc(features_flat)
        
        # Decode
        z_projected = self.decoder_fc(latent_z)
        z_reshaped = z_projected.view(z_projected.size(0), 256, 4, 4)
        decoded = self.decoder_conv(z_reshaped)
        
        # Return BOTH latent space and image for the loss calculations
        return latent_z, decoded

class AdvancedLatentDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        
        # Discriminates on 1D vectors, trying to distinguish the encoder's output 
        # from a standard Normal Distribution
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, z):
        return self.discriminator(z)