import logging;LOGGER = logging.getLogger(__name__)
import os
import PIL.Image as Image
import numpy as np
import tqdm

from datasets import DatasetInterface
import kagglehub

class Dataset(DatasetInterface):
    def __init__(self,limit:int,starting_index:int) -> None:
        self.path:str|None = None
        self.length:int = -1
        self.limit:int = limit
        self.starting_index = starting_index
    
    def transform(self,image:Image.Image) -> np.ndarray:
        x = image.convert("L")
        x = x.resize( (64,64) )
        x = np.array( x, np.float32 )
        x = x.flatten()
        x = x.clip(0,255)
        x = x / 255.0
        return x

    def download(self) -> None:
        self.path = kagglehub.dataset_download("ashwingupta3012/human-faces",force_download=False)
    
    def pre_load(self) -> None:
        
        if ( self.path == None ):
            LOGGER.critical("Dataset path was None, should be a string pointing towards the database")
            return
        
        folder = f"{self.path}\\Humans\\"
        if ( not os.path.exists(folder) ):
            LOGGER.critical("Dataset format not excpected, no ")
            return
        
        # Listt  all images in teh datset
        image_names = os.listdir(f"{folder}")
        image_paths = [ f"{folder}{image_name}" for image_name in image_names if (image_name.endswith(".png")) or (image_name.endswith(".jpg")) or (image_name.endswith(".jpeg")) ][self.starting_index:self.limit+self.starting_index]
        self.length = len(image_paths)

        # now load the dataset into memory
        self.data_cache = np.zeros( shape=(self.length,64*64) , dtype=np.float32 )

        for index,image_path in enumerate(tqdm.tqdm( image_paths, "Loading images: " )):
            x = self.transform( Image.open(image_path) )
            self.data_cache[index] = x

    def __len__(self):
        return self.length