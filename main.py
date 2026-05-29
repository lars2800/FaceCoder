import logging; LOGGER = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from devices import DEVICE
import models
import datasets
import training

torch.set_num_threads(16)
            
class Reviewer:
    def __init__(self, num_visualizations: int = 8) -> None:
        self.num_visualizations = num_visualizations
        
        self.model = models.SimpleAdversarialNetwork()
        
        dataset = datasets.TorchDataset("ashwingupta3012_human_faces", limit=num_visualizations, starting_index=4096)
        self.view_loader = DataLoader(dataset, batch_size=self.num_visualizations, shuffle=True, num_workers=0)

    def review(self, filepath: str = "checkpoints/model1.pth") -> None:
        self.model.load(filepath)

        self.model.model_auto_encoder.eval()
        
        batch_images = next(iter(self.view_loader))
        batch_images = batch_images.view(-1, 1, 64, 64).to(DEVICE)
        
        with torch.no_grad():
            reconstructed_images = self.model.model_auto_encoder(batch_images)
            
        originals = batch_images.cpu().numpy()
        reconstructions = reconstructed_images.cpu().numpy()
        
        fig, axes = plt.subplots(2, self.num_visualizations, figsize=(self.num_visualizations * 2, 7))
        
        for i in range(self.num_visualizations):
            # Row 1: Originals
            ax_orig = axes[0, i]
            ax_orig.imshow(originals[i, 0], cmap='gray')
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_title("Originals", fontsize=12, fontweight='bold', loc='left')
                
            # Row 2: Autoencoded
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
        training.SimpleAdversarialTrainer(
            epochs=30, # 100 epochs = 10min ( 4096 size )
            datset_size=4096
        ).train()
    
    if "view" in sys.argv[1:]:
        Reviewer().review()


if __name__ == "__main__":
    main()