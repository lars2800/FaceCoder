import torch

DEVICE = "cpu"

if ( torch.cuda.is_available() ):
    DEVICE = "cuda"

else:
    try:
        import torch_directml

        DEVICE = torch_directml.get_device( torch_directml.default_device() )
    except ImportError:
        pass