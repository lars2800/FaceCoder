import logging;LOGGER = logging.getLogger(__name__)
import torch
import os

torch.set_num_threads(16)

def train():

    # Create folder for saving chekpoints
    os.makedirs("checkpoints", exist_ok=True)

    # Create a neural network
    import networks
    model = networks.ImageAutoencoder( 64 * 64, 256, 8 )

    # Create a dataset
    import datasets
    dataset = datasets.TorchDataset("ashwingupta3012_human_faces",limit=512)
    
    # Start training
    import tqdm
    EPOCHS = 10
    for epoch in range(EPOCHS):

        running_loss = 0.0

        for image in tqdm.tqdm(dataset,f"Epoch {epoch}/{EPOCHS} progress: "):

            # Check how good our current model is
            out = model(image)
            loss = model.criterion(out,image)

            # Improve it
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # Save some statistics
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataset)
        LOGGER.info(f"Epoch {epoch}/{EPOCHS} loss: {avg_loss}")
    
    # Save
    checkpoint_path = f"checkpoints/autoencoder_{model.inSizeVar}_{model.numLayersVar}_{model.bottleneckSizeVar}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    
    train()


if __name__ == "__main__":
    main()