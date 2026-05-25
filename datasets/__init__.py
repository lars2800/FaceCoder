#
# How to create a datset?
# 1. Create a file with the name in datasets/ ex. 'datasets/ashwingupta3012_human_faces.py'
# 2. from datasets import DatasetInterface
# 3. create a class called 'Dataset', inherted from DatasetInterface
# 4. implement methods
# 5. use using TorchDataset(datasetname)
#

import torch
import logging;LOGGER = logging.getLogger(__name__)
import numpy as np
from torch.utils.data import Dataset as TorchDatasetInterface
from abc import ABC, abstractmethod


class TorchDataset(TorchDatasetInterface):
    def __init__(self, dataset: str, **kwargs) -> None:
        super().__init__()
        self.dataset: DatasetInterface = __import__(f"datasets.{dataset}", fromlist=['']).Dataset(**kwargs)
        self.dataset.download()
        self.dataset.pre_load()
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        
        # Get the numpy array from your underlying dataset
        np_data = self.dataset.__getitem__(idx)
        
        # Convert it to a PyTorch float tensor right here
        return torch.from_numpy(np_data).float()


class DatasetInterface(ABC):
    def __init__(self) -> None:
        self.data_cache: np.ndarray | None = None
    
    def __getitem__(self, key) -> np.ndarray:
        if self.data_cache is None:
            LOGGER.critical("data_cache has not been initialized!")
            return np.array([])
        
        return self.data_cache[key]
    
    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def pre_load(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass