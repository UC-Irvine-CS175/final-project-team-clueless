"""
This module contains the PyTorch Lightning BPSMouseDataModule to use with the PyTorch
BPSMouseDataset class. The LightningDataModule is a way to organize the training,
validation, and testing data splits and make them accessible to the model without
the overhead of creating boilerplate code for each dataset. The BPSMouseDataModule 
is a derived class from the PyTorch Lightning LightningModule class and is used to
define the training and validation datasets and dataloaders.

For detail on the PyTorch Lightning LightningDataModule class, see the documentation:
https://lightning.ai/docs/pytorch/stable/data/datamodule.html#using-a-datamodule
"""
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl

import pandas as pd
import numpy as np

import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

import sys
sys.path.append(str(root))
from src.data_utils import save_tiffs_local_from_s3
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from lightly.data import SimCLRCollateFunction
from src.dataset.simclr_dataloader.bps_dataset import BPSMouseDataset

from src.dataset.simclr_dataloader.augmentation import (
    TransformsSimCLR
)

# To define the PyTorch Lightning DataModule, we need to define a class that inherits from the
# PyTorch Lightning DataModule class. 
class BPSDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv_file: str,
                 train_dir: str,
                 val_csv_file: str,
                 val_dir: str,
                 transform: transforms = None,
                 test_csv_file: str = None,
                 test_dir: str = None,
                 file_on_prem: bool = True,
                 batch_size: int = 4,
                 num_workers: int = 2,
                 collate_fn: SimCLRCollateFunction = None,
                 meta_csv_file: str = None,
                 meta_root_dir: str = None,
                 s3_client: boto3.client = None,
                 s3_path: str = None,
                 bucket_name: str = None):
        """
        PyTorch Lightning DataModule for the BPS microscopy data.

        Args:
            train_csv_file (str): The name of the csv file containing the training data.
            train_dir (str): The directory where the training data is stored.
            val_csv_file (str): The name of the csv file containing the validation data.
            val_dir (str): The directory where the validation data is stored.
            resize_dims (tuple): The dimensions to resize the images to during Transform.
            test_csv_file (str): The name of the csv file containing the test data.
            test_dir (str): The directory where the test data is stored.
            file_on_prem (bool): Whether the data is stored on-prem or in the cloud 
                                 (needed by BPSDataset)
            batch_size (int): The batch size to use for the DataLoader.
            num_workers (int): The number of workers to use for the DataLoader.
            meta_csv_file (str): The name of the csv file containing the all the metadata
                                (needed by BPSDataset when file_on_prem = False)
            meta_root_dir (str): The directory where the metadata is stored
                                (needed by BPSDataset when file_on_prem = False)
        """
        super().__init__()
        self.train_csv = train_csv_file
        self.train_dir = train_dir
        self.val_csv = val_csv_file
        self.val_dir = val_dir
        self.test_csv = test_csv_file
        self.test_dir = test_dir
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.s3_path = s3_path
        self.on_prem = file_on_prem
        self.meta_csv = meta_csv_file
        self.meta_dir = meta_root_dir
        self.transform = transform
        self.on_prem = file_on_prem
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self) -> None:
        """
        Download data if needed. This method is called only from a single CPU.
        """
        # Download the csv files from S3 using the save_tiffs_local_from_s3 function
        # defined in src/data_utils.py assuming that the csv files specific to the
        # train, val, and test splits have already been created and are stored locally.

        # call the save_tiffs_local_from_s3 function to download tiffs from the
        # train_csv file.
        save_tiffs_local_from_s3(s3_client=self.s3_client,
                                 bucket_name=self.bucket_name,
                                 s3_path=self.s3_path,
                                 local_fnames_meta_path=f"{self.train_dir}/{self.train_csv}",
                                 save_file_path=self.train_dir)
        # call the save_tiffs_local_from_s3 function to download tiffs from the
        # val_csv file.        
        save_tiffs_local_from_s3(s3_client=self.s3_client,
                                 bucket_name=self.bucket_name,
                                 s3_path=self.s3_path,
                                 local_fnames_meta_path=f"{self.val_dir}/{self.val_csv}",
                                 save_file_path=self.val_dir)
        # This if statement is provided because the BPS dataset has held out the test files.
        if self.test_csv is not None:
            # call the save_tiffs_local_from_s3 function to download tiffs from the
            # test_csv file assuming that it is available locally.
            save_tiffs_local_from_s3(s3_client=self.s3_client,
                                    bucket_name=self.bucket_name,
                                    s3_path=self.s3_path,
                                    local_fnames_meta_path=f"{self.test_dir}/{self.test_csv}",
                                    save_file_path=self.test_dir)
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders. Requires that train,
        val, and test csv files be stored locally. Image tiffs will be stored
        in the same directory as the csv files.
        """
        if stage == "train":
            # Create the BPSMouseDataset object for the training data.
            self.bps_training = BPSMouseDataset(meta_csv_file= self.train_csv,
                                                meta_root_dir= self.train_dir,
                                                transform=self.transform,
                                                file_on_prem=self.on_prem)
        if stage == "validate":
            # Create the BPSMouseDataset object for the validation data.
            self.bps_validate = BPSMouseDataset(meta_csv_file=self.val_csv,
                                                meta_root_dir=self.val_dir,
                                                transform=self.transform,
                                                file_on_prem=self.on_prem)
            
        if stage == "test":
            # Create the BPSMouseDataset object for the test data.
            self.bps_test = BPSMouseDataset(meta_csv_file=self.test_csv,
                                                meta_root_dir=self.test_dir,
                                                transform=self.transform,
                                                file_on_prem=self.on_prem)
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns the training dataloader.
        """
        return DataLoader(self.bps_training,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the validation dataloader.
        """
        return DataLoader(self.bps_validate,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the test dataloader. In our case, we will only use the val_dataloader
        since NASA GeneLab has not released the test set.
        """
        return DataLoader(self.bps_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)

    

def main():
    """main function to test PyTorch Dataset class"""
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname = "meta.csv"


    data_dir = root / 'data'
    # testing get file functions from s3
    local_train_dir = data_dir / 'processed'

    #testing PyTorch Lightning DataModule class ####
    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    train_dir = data_dir / 'processed'
    validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    validation_dir = data_dir / 'processed'
    my_transform = TransformsSimCLR(300, 254)
    bps_dm = BPSDataModule(train_csv_file=train_csv_file,
                           train_dir=train_dir,
                           val_csv_file=validation_csv_file,
                           val_dir=validation_dir,
                           transform=my_transform,
                           meta_csv_file = s3_meta_fname,
                           meta_root_dir=s3_path,
                           s3_client= s3_client,
                           bucket_name=bucket_name,
                           s3_path=s3_path,
                           )
    ##### UNCOMMENT THE LINE BELOW TO DOWNLOAD DATA FROM S3!!! #####
    #bps_dm.prepare_data()
    ##### WHEN YOU ARE DONE REMEMBER TO COMMENT THE LINE ABOVE TO AVOID
    ##### DOWNLOADING THE DATA AGAIN!!! #####
    bps_dm.setup(stage='train')

    for batch_idx, batch in enumerate(bps_dm.train_dataloader()):
        img0, img1, l1, l2, fname = batch
        print(batch_idx, img0.shape, img1.shape, l1, l2, fname)
        break


if __name__ == "__main__":
    main()