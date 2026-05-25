import logging; LOGGER = logging.getLogger(__name__)
import torch
import os
import tqdm
from torch.utils.data import DataLoader

from devices import DEVICE
torch.set_num_threads(16)

def train():

    # Create folder for saving chekpoints
    os.makedirs("checkpoints", exist_ok=True)

    # Create a neural network
    import networks
    model = networks.ImageAutoencoder( 64 * 64, 256, 8 ).to(DEVICE)

    # Create a dataset
    import datasets
    dataset = datasets.TorchDataset("ashwingupta3012_human_faces",limit=512,starting_index=0)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Start training
    EPOCHS = 10
    model.train()
    for epoch in range(EPOCHS):

        running_loss = 0.0

        for batch_images in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            # Flatten images to (batch_size, 4096)
            batch_images = batch_images.view(batch_images.size(0), -1).to(DEVICE)

            # Forward pass
            out = model(batch_images)
            loss = model.criterion(out, batch_images)

            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item() * batch_images.size(0)
        
        avg_loss = running_loss / len(dataset)
        LOGGER.info(f"Epoch {epoch+1}/{EPOCHS} loss: {avg_loss}")
    
    # Save
    checkpoint_path = f"checkpoints/autoencoder_{model.inSizeVar}_{model.numLayersVar}_{model.bottleneckSizeVar}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info(f"Saved checkpoint to {checkpoint_path}")

def review():
    pass

def main():
    import sys

    logging.basicConfig(level=logging.INFO)
    
    # Yes I know, but it's just for testing, the sys arguments don't matter that much for now
    if "train" in sys.argv[1:]:
        train()
    
    if "view" in sys.argv[1:]:
        review()


if __name__ == "__main__":
    main()