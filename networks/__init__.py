import logging
import torch

LOGGER = logging.getLogger(__name__)

class ImageAutoencoder(torch.nn.Module):
    def __init__(self, inSize: int, bottleneckSize: int, num_layers: int = 3):
        """
        An efficient autoencoder that dynamically calculates intermediate layer sizes 
        logarithmically to avoid creating too many unnecessary layers.
        
        Args:
            inSize (int): Flattened input size (e.g., width * height).
            bottleneckSize (int): Latent space / compression size.
            num_layers (int): Exact number of reduction steps to reach the bottleneck.
        """
        super().__init__()

        self.inSizeVar = inSize
        self.bottleneckSizeVar = bottleneckSize
        self.numLayersVar = num_layers

        # --- Encoder ---
        LOGGER.info("Creating encoding layers")
        encoderLayers = []
        lastSize = inSize
        
        # Calculate intermediate sizes smoothly between inSize and bottleneckSize
        # Using linear interpolation in log-space ensures smooth geometric downsizing
        for i in range(1, num_layers):
            # Calculate size at this step
            fraction = i / num_layers
            # Geometric step formula:
            layerSize = int(inSize * ((bottleneckSize / inSize) ** fraction))
            
            encoderLayers.extend([
                torch.nn.Linear(lastSize, layerSize),
                torch.nn.ReLU(),
            ])
            lastSize = layerSize
            
        # Final bridge to bottleneck
        encoderLayers.extend([
            torch.nn.Linear(lastSize, bottleneckSize),
            torch.nn.ReLU(),
        ])
        
        self.encoder = torch.nn.Sequential(*encoderLayers)

        # --- Decoder ---
        LOGGER.info("Creating decoding layers")
        decoderLayers = []
        lastSize = bottleneckSize
        
        # Mirror the encoder sizes in reverse order
        # We grab the shapes from the encoder to perfectly match its structure
        encoder_linear_layers = [l for l in self.encoder if isinstance(l, torch.nn.Linear)]
        
        for enc_layer in reversed(encoder_linear_layers[1:]):
            # The output size of an encoder layer becomes the input size of a decoder layer
            layerSize = enc_layer.in_features 
            decoderLayers.extend([
                torch.nn.Linear(lastSize, layerSize),
                torch.nn.ReLU(),
            ])
            lastSize = layerSize

        # Final reconstruction layer back to the original image dimensions
        decoderLayers.extend([
            torch.nn.Linear(lastSize, inSize),
            torch.nn.Sigmoid(), # Best if image pixels are normalized between 0 and 1
        ])
        
        self.decoder = torch.nn.Sequential(*decoderLayers)
        LOGGER.info(f"Done creating autoencoder layers. Encoder layers: {len(self.encoder)//2}, Decoder layers: {len(self.decoder)//2}")

        # Criterion & Optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded