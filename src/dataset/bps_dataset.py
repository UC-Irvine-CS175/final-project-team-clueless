"""
This module contains the BPSMouseDataset class which is a subclass of torch.utils.data.Dataset.
"""
from src.dataset.augmentation import (
    NormalizeBPS,
    ResizeBPS,
    VFlipBPS,
    HFlipBPS,
    RotateBPS,
    RandomCropBPS,
    ToTensor
)

import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torchvision import transforms, utils

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
from pyprojroot import here

root = here()

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import io
from io import BytesIO
from PIL import Image
from .. import data_utils

""" 
Note about Tensor:
        PyTorch expects tensors to have the following dimensions:
        (batch_size, channels, height, width)
        A numpy array has the following dimensions:
        (height, width, channels)
"""


class BPSMouseDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for the BPS microscopy data.

    args:
        meta_csv_file (str): name of the metadata csv file
        meta_root_dir (str): path to the metadata csv file
        bucket_name (str): name of bucket from AWS open source registry.
        transform (callable, optional): Optional transform to be applied on a sample.

    attributes:
        meta_df (pd.DataFrame): dataframe containing the metadata
        bucket_name (str): name of bucket from AWS open source registry.
        train_df (pd.DataFrame): dataframe containing the metadata for the training set
        test_df (pd.DataFrame): dataframe containing the metadata for the test set
        transform (callable): The transform to be applied on a sample.

    raises:
        ValueError: if the metadata csv file does not exist
    """

    def __init__(
            self,
            meta_csv_file:str,
            meta_root_dir:str,
            s3_client: boto3.client = None,
            bucket_name: str = None,
            transform=None,
            file_on_prem:bool = True):

        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.transform = transform
        self.file_on_prem = file_on_prem
        
        if self.file_on_prem:
            # formulate the full path to metadata csv file
            self.meta_csv_path = os.path.join(meta_root_dir, meta_csv_file)

            if not os.path.exists(self.meta_csv_path):
                raise ValueError(f"Metadata csv file {self.meta_csv_path} does not exist")
            
            self.meta_df = pd.read_csv(self.meta_csv_path)
        else:
            # if the file is not on the local file system, use the get_bytesio_from_s3 function
            # to fetch the file as a BytesIO object, else read the file from the local file system.

            if not self.s3_client or not self.bucket_name:
                raise ValueError("S3 client and bucket name must be provided for S3-based file access.")
            
            meta_key = os.path.join(meta_root_dir, meta_csv_file)
            meta_bytesio = get_bytesio_from_s3(self.s3_client, self.bucket_name, meta_key)
            self.meta_df = pd.read_csv(meta_bytesio)


    def __len__(self):
        """
        Returns the number of images in the dataset.

        returns:
          len (int): number of images in the dataset
        """

        return len(self.meta_df)

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding label for a given index.

        Args:
            idx (int): index of the image to fetch

        Returns:
            img_tensor (torch.Tensor): tensor of image data
            label (int): label of image
        """

        # get the bps image file name from the metadata dataframe at the given index
        img_file = self.meta_df.iloc[idx]['filename']
        img_label = self.meta_df.iloc[idx]['particle_type']

        # formulate path to image given the root directory (note meta.csv is in the
        # same directory as the images)
        img_path = os.path.join(os.path.dirname(self.meta_csv_path), img_file)

        # If on_prem is False, then fetch the image from s3 bucket using the get_bytesio_from_s3
        # function, get the contents of the buffer returned, and convert it to a  numpy array
        # with datatype unsigned 16 bit integer used to represent microscopy images.
        # If on_prem is True load the image from local. 

        if not self.file_on_prem:
            img_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=img_path)
            img_bytes = img_obj['Body'].read()
            img = np.frombuffer(img_bytes, dtype=np.uint16).reshape(1024, 1024)
        else:
            img = np.array(Image.open(img_path).convert("L"), dtype=np.uint16)

        # apply tranformation if available
        if self.transform:
            img = self.transform(img)

        # check if image is a torch.Tensor
        if not torch.is_tensor(img):
            # convert np array to img
            img = Image.fromarray(img)

            # convert the image to a torch.Tensor
            img = transforms.ToTensor()(img)

        # return the image and associated label
        return img, img_label


def show_label_batch(image: torch.Tensor, label: str):
    """Show image with label for a batch of samples."""
    images_batch, label_batch = \
            image, label
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    # grid is a 4 dimensional tensor (channels, height, width, number of images/batch)
    # images are 3 dimensional tensor (channels, height, width), where channels is 1
    # utils.make_grid() takes a 4 dimensional tensor as input and returns a 3 dimensional tensor
    # the returned tensor has the dimensions (channels, height, width), where channels is 3
    # the returned tensor represents a grid of images
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.savefig('test_grid_1_batch.png')



def main():
    """main function to test PyTorch Dataset class (Make sure the directory structure points to where the data is stored)"""
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_meta_csv_path = f"{s3_path}/meta.csv"

    #### testing get file functions from s3 ####

    local_file_path = "../data/raw"
    local_train_csv_path = "../data/processed/meta_dose_hi_hr_4_post_exposure_train.csv"

    print(root)

    #### testing dataset class ####
    train_csv_path = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    training_bps = BPSMouseDatasetLocal(train_csv_path, '../data/processed', transform=None, file_on_prem=True)
    print(training_bps.__len__())
    print(training_bps.__getitem__(0))

    transformed_dataset = BPSMouseDataset(train_csv_path,
                                           '../data/processed',
                                           transform=transforms.Compose([
                                               NormalizeBPS(),
                                               ResizeBPS(224, 224),
                                               VFlipBPS(),
                                               HFlipBPS(),
                                               RotateBPS(90),
                                               RandomCropBPS(200, 200),
                                               ToTensor()
                                            ]),
                                            file_on_prem=True
                                           )

    # Use Dataloader to package data for batching, shuffling, 
    # and loading in parallel using multiprocessing workers
    # Packaging is image, label
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)

    for batch, (image, label) in enumerate(dataloader):
        print(batch, image, label)

        if batch == 5:
            show_label_batch(image, label)
            print(image.shape)
            break

if __name__ == "__main__":
    main()
#The PyTorch Dataset class is an abstract class that is used to provide an interface for accessing all the samples
# in your dataset. It inherits from the PyTorch torch.utils.data.Dataset class and overrides two methods:
# __len__ and __getitem__. The __len__ method returns the number of samples in the dataset and the __getitem__

