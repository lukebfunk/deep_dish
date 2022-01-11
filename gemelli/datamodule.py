import pandas as pd
from pytorch_lightning import LightningDataModule

from deep_dish.datasets import CellPatchesDataset
from deep_dish.dataloader import BalancedDataLoader

class DataModule(LightningDataModule):
    def __init__(self,train_csv,val_csv,test_csv):
        super().__init__()
        self.train_csv=train_csv
        self.val_csv=val_csv
        self.test_csv=test_csv
        
    def train_dataloader(self):
        train_split = CellPatchesDataset(self.train_csv)
        return BalancedDataLoader(train_split)
        
    def val_dataloader(self):
        val_split = CellPatchesDataset(self.val_csv)
        return BalancedDataLoader(val_split)

    def test_dataloader(self):
        test_split = CellPatchesDataset(self.test_csv)
        return BalancedDataLoader(test_split)
