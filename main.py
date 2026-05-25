import logging; LOGGER = logging.getLogger(__name__)
import torch
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils

torch.set_num_threads(16)

def train():

    # Create folder for saving chekpoints
    os.makedirs("checkpoints", exist_ok=True)

    # Create a neural network
    import networks
    model = networks.ImageAutoencoder( 64 * 64, 256, 8 )

    # Create a dataset
    import datasets
    dataset = datasets.TorchDataset("ashwingupta3012_human_faces",limit=512,starting_index=0)
    
    # Start training
    import tqdm
    EPOCHS = 10
    for epoch in range(EPOCHS):

        running_loss = 0.0

        for image in tqdm.tqdm(dataset,f"Epoch {epoch+1}/{EPOCHS} progress: "):

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
        LOGGER.info(f"Epoch {epoch+1}/{EPOCHS} loss: {avg_loss}")
    
    # Save
    checkpoint_path = f"checkpoints/autoencoder_{model.inSizeVar}_{model.numLayersVar}_{model.bottleneckSizeVar}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info(f"Saved checkpoint to {checkpoint_path}")

def review():
    # Create a neural network
    import networks
    model = networks.ImageAutoencoder( 64 * 64, 256, 8 )

    # Load the trained weights if they exist
    checkpoint_path = f"checkpoints/autoencoder_{model.inSizeVar}_{model.numLayersVar}_{model.bottleneckSizeVar}.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        LOGGER.info(f"Loaded weights from {checkpoint_path}")
    else:
        LOGGER.warning("No checkpoint found! Displaying outputs from an untrained model.")

    # Set model to evaluation mode
    model.eval()

    # Create a dataset (using 10 validation images)
    import datasets
    dataset = datasets.TorchDataset("ashwingupta3012_human_faces", limit=10, starting_index=512)
    
    originals = []
    reconstructions = []

    # Run images through the model without calculating gradients
    with torch.no_grad():
        for image in dataset:
            # Pass through the model
            reconstructed_image = model(image)
            
            # Reshape from flat vectors back to (Channels, Height, Width) -> (1, 64, 64)
            # Adjust the shapes if your dataset uses 3 channels (RGB) instead of 1 (Grayscale)
            originals.append(image.view(1, 64, 64))
            reconstructions.append(reconstructed_image.view(1, 64, 64))

    # Stack lists into batches: shape (10, 1, 64, 64)
    orig_tensor = torch.stack(originals)
    recon_tensor = torch.stack(reconstructions)

    # Combine both rows: Top row is original, Bottom row is model output
    comparison_tensor = torch.cat([orig_tensor, recon_tensor], dim=0)

    # Create a grid layout (nrow=10 means 10 images per row)
    grid = vutils.make_grid(comparison_tensor, nrow=10, normalize=True)

    # Convert PyTorch tensor (C, H, W) to Matplotlib format (H, W, C)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Plot the result
    plt.figure(figsize=(12, 4))
    plt.title("Original Images (Top) vs Model Reconstructions (Bottom)")
    plt.imshow(grid_np, cmap='gray' if grid_np.shape[2] == 1 else None)
    plt.axis('off')
    
    # Show the plot window, or uncomment the next line to save it as an image file
    plt.show()
    # plt.savefig("comparison_results.png", bbox_inches='tight')


def main():
    logging.basicConfig(level=logging.INFO)
    
    review()


if __name__ == "__main__":
    main()