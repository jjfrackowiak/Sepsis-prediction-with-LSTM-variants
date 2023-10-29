from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from tqdm import tqdm
import pickle
from data_processing import SepsisDataset

class SepsisDataloader(pl.LightningDataModule):
    def __init__(self, data_dir, window_size, batch_size, num_workers=0, device = 'cpu'):
        super().__init__()
        self.device = device
        self.data_dir=data_dir
        self. window_size=window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = None
        self.val_set = None

    def setup(self):
        # Instantiate datasets
        self.train_set = SepsisDataset(data_dir = self.data_dir, is_train=True, window_size=self.window_size, device = self.device)
        self.val_set = SepsisDataset(data_dir = self.data_dir, is_train=False, window_size=self.window_size, device = self.device)
        self.train_set.setup()
        self.val_set.scaler = self.train_set.scaler
        self.val_set.setup()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
