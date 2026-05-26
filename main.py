import logging; LOGGER = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from devices import DEVICE
import networks
import datasets
import misc

torch.set_num_threads(16)

class Trainer:
    def __init__(self) -> None:
        #
        # Settings
        #
        self.EPOCHS = 100
        self.adv_loss_mult = 0.00125

        #
        # Create neural networks and optimizers
        #

        self.auto_encoder_model   = networks.SimpleImageAutoEncoder(in_channels=1).to(DEVICE)
        self.discriminator_model  = networks.SimpleImageDiscirminator(in_channels=1).to(DEVICE)

        self.optimizer_auto_encoder = torch.optim.Adam(self.auto_encoder_model.parameters(), lr=0.0002125)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator_model.parameters(), lr=0.0002125)
        
        self.reconstruction_criterion = torch.nn.MSELoss()
        self.adversarial_criterion = torch.nn.BCELoss()

        #
        # Load datset
        #

        dataset = datasets.TorchDataset("ashwingupta3012_human_faces", limit=4096, starting_index=0)
        self.train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    def train(self) -> None:
        #
        # Main train loop
        #

        for epoch in range(self.EPOCHS):
            timer = misc.Timer.begin()
            self.train_auto_encoder_epoch(epoch)
            self.train_discriminator_epoch(epoch)
            LOGGER.info(f"Epoch finsihed in: {timer}")
        
        self.save("checkpoints/checkpoint.pth")
    
    def freezer(self,fase_is_encoder:bool) -> None:
        #
        # Freezes and unfreesez parameters as needed
        #

        if ( fase_is_encoder ):
            # Freeze discriminator / Unfreeze autoencoder
            self.discriminator_model.eval()

            for param in self.discriminator_model.parameters():
                param.requires_grad = False
            for param in self.discriminator_model.parameters():
                param.requires_grad = True
            
            self.auto_encoder_model.train(True)
        
        else:
            self.auto_encoder_model.eval()

            for param in self.auto_encoder_model.parameters():
                param.requires_grad = False
            for param in self.auto_encoder_model.parameters():
                param.requires_grad = True
            
            self.discriminator_model.train(True)

    def train_auto_encoder_epoch(self, epoch: int):
        self.freezer( fase_is_encoder=True )

        running_loss = 0.0
        
        for batch_images in self.train_loader:
            batch_images = batch_images.view(-1, 1, 64, 64).to(DEVICE)

            # Forward pass
            out = self.auto_encoder_model(batch_images)

            # Calculate how good it was
            recon_loss = self.reconstruction_criterion(out, batch_images) 

            discriminator_predictions = self.discriminator_model(out) 
            real_labels = torch.ones_like(discriminator_predictions).to(DEVICE) # this is what we try to achieve all ones
            adv_loss = self.adversarial_criterion(discriminator_predictions, real_labels) 

            loss = recon_loss + (self.adv_loss_mult * adv_loss) # We need a mix between the 2 criteroens

            # Backward pass
            self.optimizer_auto_encoder.zero_grad()
            loss.backward()
            self.optimizer_auto_encoder.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(self.train_loader)
        LOGGER.info(f"AutoEncoder - Epoch {epoch+1}/{self.EPOCHS} avg. loss: {avg_loss:.4f}")

    def train_discriminator_epoch(self, epoch: int):
        self.freezer(fase_is_encoder=False)
            
        running_loss = 0.0
        
        for batch_images in self.train_loader:
            batch_images = batch_images.view(-1, 1, 64, 64).to(DEVICE)

            # 1. Train on Real Images
            real_predictions = self.discriminator_model(batch_images)
            real_labels = torch.ones_like(real_predictions).to(DEVICE)
            loss_real = self.adversarial_criterion(real_predictions, real_labels)

            # 2. Train on Fake Images
            fake_images = self.auto_encoder_model(batch_images).detach()
            fake_predictions = self.discriminator_model(fake_images)
            fake_labels = torch.zeros_like(fake_predictions).to(DEVICE)
            loss_fake = self.adversarial_criterion(fake_predictions, fake_labels)

            loss = loss_real + loss_fake

            # Backward pass using the persistent optimizer
            self.optimizer_discriminator.zero_grad()
            loss.backward()
            self.optimizer_discriminator.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        LOGGER.info(f"Discriminator - Epoch {epoch+1}/{self.EPOCHS} avg. loss: {avg_loss:.4f}")
    
    def save(self, filepath: str = "checkpoint.pth") -> None:
        #
        # Saving the model
        #
        
        LOGGER.info(f"Saving model checkpoints to {filepath}...")
        
        checkpoint = {
            "autoencoder_state_dict": self.auto_encoder_model.state_dict(),
            "discriminator_state_dict": self.discriminator_model.state_dict()
        }

        try:
            torch.save(checkpoint, filepath)
            LOGGER.info("Checkpoints saved successfully.")
        
        except Exception as e:
            LOGGER.error(f"Failed to save checkpoints: {e}")

class Reviewer:
    def __init__(self, num_visualizations: int = 8) -> None:
        self.num_visualizations = num_visualizations
        
        # Re-initialize the same AutoEncoder structure
        self.autoEncoder = networks.SimpleImageAutoEncoder(in_channels=1).to(DEVICE)
        
        # Load dataset to grab a validation/review batch
        dataset = datasets.TorchDataset("ashwingupta3012_human_faces", limit=num_visualizations, starting_index=4096)
        self.view_loader = DataLoader(dataset, batch_size=self.num_visualizations, shuffle=True, num_workers=0)

    def review(self, filepath: str = "checkpoints/checkpoint.pth") -> None:
        LOGGER.info(f"Loading weights for evaluation from {filepath}...")
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.autoEncoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        except Exception as e:
            LOGGER.error(f"Could not load checkpoint: {e}")
            return

        # Put the model into evaluation mode
        self.autoEncoder.eval()
        
        # Get one batch of images
        batch_images = next(iter(self.view_loader))
        batch_images = batch_images.view(-1, 1, 64, 64).to(DEVICE)
        
        # Generate predictions without tracking gradients
        with torch.no_grad():
            reconstructed_images = self.autoEncoder(batch_images)
            
        # Move tensors back to CPU and convert to numpy for plotting
        originals = batch_images.cpu().numpy()
        reconstructions = reconstructed_images.cpu().numpy()
        
        # Setup matplotlib figure: 2 rows (Originals on top, Autoencoded on bottom)
        fig, axes = plt.subplots(2, self.num_visualizations, figsize=(self.num_visualizations * 2, 5))
        
        for i in range(self.num_visualizations):
            # --- Top Row: Original Images ---
            ax_orig = axes[0, i]
            ax_orig.imshow(originals[i, 0], cmap='gray')
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_title("Originals", fontsize=12, fontweight='bold', loc='left')
                
            # --- Bottom Row: Autoencoded Images ---
            ax_recon = axes[1, i]
            ax_recon.imshow(reconstructions[i, 0], cmap='gray')
            ax_recon.axis('off')
            if i == 0:
                ax_recon.set_title("Autoencoded", fontsize=12, fontweight='bold', loc='left')
                
        plt.tight_layout()
        plt.show()

def main():
    import sys

    logging.basicConfig(level=logging.INFO)
    
    # Yes I know, but it's just for testing, the sys arguments don't matter that much for now
    if "train" in sys.argv[1:]:
        Trainer().train()
    
    if "view" in sys.argv[1:]:
        Reviewer().review()


if __name__ == "__main__":
    main()