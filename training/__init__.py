import logging; LOGGER = logging.getLogger(__name__)
from torch.utils.data import DataLoader

from devices import DEVICE
import datasets
import misc

from abc import ABC, abstractmethod

class ImageTrainertemplate(ABC):
    def __init__(self,epochs:int=25,datset_size:int=1024,batch_size:int=64,learn_rate:float=0.0002) -> None:

        self.datset_size = datset_size
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.EPOCHS = epochs

        self.load_dataset(
            num_images=self.datset_size,
            batch_size=self.batch_size
        )
        self.load_models()
        self.create_optimizers_and_critereons()
    
    @abstractmethod
    def create_optimizers_and_critereons(self) -> None:
        pass
    
    @abstractmethod
    def train_epoch(self,epoch:int) -> None:
        pass

    @abstractmethod
    def load_models(self) -> None:
        pass

    def train(self):

        # Create a fancy anner
        banner_str = f"[ Training: \033[0;36m{self.EPOCHS}\033[0;37m EPOCHS, \033[0;36m{self.datset_size}\033[0;37m images, Learnrate: \033[0;36m{self.learn_rate}\033[0;37m ]\n"
        print("\n",end="")
        print(f"[{'-'*(len(banner_str)-45)}]\n",end="")
        print(f"[{' '*(len(banner_str)-45)}]\n",end="")
        print(banner_str,end="")
        print(f"[{' '*(len(banner_str)-45)}]\n",end="")
        print(f"[{'-'*(len(banner_str)-45)}]\n",end="")
        print("\n",end="")

        # Create a progress bar
        self.epoch_bar = misc.RainbowProgressBar(range(self.EPOCHS), bar_format='[ {remaining} - ({rate_fmt} ) -> {elapsed} ] Epoch {n_fmt}/{total_fmt}{postfix} |{bar}|')

        for epoch in self.epoch_bar:
            if ( epoch % 100 == 99 ):
                self.model.save(log=False) 
            self.train_epoch(epoch)
        
        print("\n",end="")
        self.model.save()
    
    def transform_batch(self,batch):
        """
        Takes in a batch from the datset and converts it into something that the neural nets understand
        
        Args:
            batch: batch to transform

        Returns:
            Transformed batch
        """        
        return batch.view(-1, 1, 64, 64).to(DEVICE)
    
    def load_dataset(self,num_images:int=4096,batch_size:int=128) -> None:
        """
        Loads the datset
        """
        # Load half of the images for the auto encoder and the other half for the discriminator
        dataset = datasets.TorchDataset("ashwingupta3012_human_faces",limit=num_images,starting_index=0)
        self.data_loader  = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)