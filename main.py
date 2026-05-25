import logging;LOGGER = logging.getLogger(__name__)

def train():
    import datasets

    datasets.TorchDataset("ashwingupta3012_human_faces")

def main():
    logging.basicConfig(level=logging.INFO)
    
    train()


if __name__ == "__main__":
    main()
