# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
import os
import numpy as np
from dataclasses import dataclass
import wandb
import matplotlib.pyplot as plt
import cv2

from src.dataset.simclr_dataloader.bps_dataset import BPSMouseDataset
from src.dataset.simclr_dataloader.augmentation import TransformsSimCLR
from src.dataset.simclr_dataloader.bps_datamodule import BPSDataModule
from src.dataset.augmentation import ResizeBPS

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

import matplotlib.offsetbox as osb
import matplotlib.pyplot as plt

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp
from PIL import Image

# for clustering and 2d representations
from sklearn import random_projection

@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        acc_devices: Number of devices to use for training. Defaults to 1.
        
        device: Type of device used for training, checks for 'cuda', otherwise defaults to 'cpu'

        num_workers: Number of cpu cores dedicated to processing the data in the dataloader

        dm_stage: Set the partition of data depending to either 'train', 'val', or 'test'
                    However, our test images are not yet available.


    """
    data_dir:           str = os.path.join(root,'train')
    train_meta_fname:   str = 'meta_train.csv'
    val_meta_fname:     str = 'meta_test.csv'
    save_vis_dir:       str = os.path.join(root, 'visualizations', 'simclr_knn')
    save_models_dir:    str = os.path.join(root, 'models', 'simclr')
    batch_size:         int = 8
    max_epochs:         int = 32
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 4 # 8: change once able to train successfully
    dm_stage:           str = 'train'
    seed:               int = 1

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        #self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        x0, x1, _, _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        wandb.log({'train_loss_ssl' : loss}) 
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        return optim
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=10)
        # return [optim], [scheduler]
    
def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    labels = []
    with torch.no_grad():
        for img, _, label, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            labels.extend(torch.argmax(label, dim=1))
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames, labels

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, labels, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            #fname = os.path.join(path_to_data, filenames[neighbor_idx])
            fname = filenames[neighbor_idx]
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}, l={labels[neighbor_idx]}")
            # let's disable the axis
            plt.axis("off")
            plt.savefig(os.path.join(root, 'visualizations', 'simclr', f'knn_example_{idx}.png'))
    #plt.savefig("knn_examples.png")

def get_scatter_plot_with_thumbnails(embeddings, filenames):
        """Creates a scatter plot with image overlays."""

        # for the scatter plot we want to transform the images to a two-dimensional
        # vector space using a random Gaussian projection
        projection = random_projection.GaussianRandomProjection(n_components=2)
        embeddings_2d = projection.fit_transform(embeddings)

        # normalize the embeddings to fit in the [0, 1] square
        M = np.max(embeddings_2d, axis=0)
        m = np.min(embeddings_2d, axis=0)
        embeddings_2d = (embeddings_2d - m) / (M - m)

        
        # initialize empty figure and add subplot
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle("Scatter Plot of the BPS Dataset")
        ax = fig.add_subplot(1, 1, 1)
        # shuffle images and find out which images to show
        shown_images_idx = []
        shown_images = np.array([[1.0, 1.0]])
        iterator = [i for i in range(embeddings_2d.shape[0])]
        np.random.shuffle(iterator)
        for i in iterator:
            # only show image if it is sufficiently far away from the others
            dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
            if np.min(dist) < 2e-3:
                continue
            shown_images = np.r_[shown_images, [embeddings_2d[i]]]
            shown_images_idx.append(i)

        # plot image overlays
        for idx in shown_images_idx:
            thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
            path = os.path.join("Microscopy/train", filenames[idx])
            img = Image.open(path)
            img = cv2.resize(np.array(img), (48, 48)) #functional.resize(img, thumbnail_size)
            img = np.array(img)
            img_box = osb.AnnotationBbox(
                osb.OffsetImage(img), # osb.OffsetImage(img, cmap=plt.cm.gray_r),
                embeddings_2d[idx],
                pad=0.2,
            )
            ax.add_artist(img_box)

        # set aspect ratio
        ratio = 1.0 / ax.get_data_ratio()
        ax.set_aspect(ratio, adjustable="box")
        plt.savefig(os.path.join(root, 'visualizations', 'simclr', f'scatterplot_example.png'))


############################# Visualizing 3x3 Nearest Neighbors Images

def get_image_as_np_array(filename: str):
    """Loads the image with filename and returns it as a numpy array."""
    img = Image.open(filename)
    return np.asarray(img)


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img


def plot_nearest_neighbors_3x3(example_image: str, i: int, embeddings, filenames):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    example_idx = filenames.index(example_image)
    # get distances to the cluster center
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = os.path.join("Microscopy/train", filenames[plot_idx])
        if plot_offset == 0:
            ax.set_title(f"Example Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
        else:
            plt.imshow(get_image_as_np_array(fname))
        # let's disable the axis
        plt.axis("off")

#####################################################

def main():
    config = BPSConfig()
    pl.seed_everything(config.seed)
    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   transform=TransformsSimCLR(254, 128),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage=config.dm_stage)
    bps_datamodule.setup(stage='validate')

    
    
    # wandb.init(project="SimCLR",
    #            dir=config.save_vis_dir,
    #            config=
    #            {
    #                "architecture": "SimCLR",
    #                "dataset": "BPS Microscopy"
    #            })
    # model = SimCLR()
    # trainer = pl.Trainer(max_epochs=config.max_epochs, devices=1, accelerator=config.accelerator)
    # trainer.fit(model, bps_datamodule.train_dataloader())
    # model.eval()
    # embeddings, filenames, labels = generate_embeddings(model, bps_datamodule.val_dataloader())
    # print(f'embeddings.shape: {embeddings.shape}')
    # print(f'len(filenames): {len(filenames)}')
    # print(f'len(labels): {len(labels)}')
    # plot_knn_examples(embeddings, filenames, labels)
    # wandb.finish()

    # # You could use the pretrained model and train a classifier on top.
    # pretrained_resnet_backbone = model.backbone
    
    # # you can also store the backbone and use it in another code
    # state_dict = {"resnet18_parameters": pretrained_resnet_backbone.state_dict()}
    # torch.save(state_dict, os.path.join(config.save_models_dir, "model.pth"))

    ###################################################################################

    # Create an instance of the SimCLR model
    model = SimCLR()

    # Load the saved state dictionary
    state_dict = torch.load(os.path.join(config.save_models_dir, "model.pth"))

    print("LOADING MODEL...")
    # Load the backbone parameters into the SimCLR model
    model.backbone.load_state_dict(state_dict["resnet18_parameters"])
    print("MODEL LOADED")

    embeddings, filenames, labels = generate_embeddings(model, bps_datamodule.val_dataloader())
    print(f'embeddings.shape: {embeddings.shape}')
    print(f'len(filenames): {len(filenames)}')
    print(f'len(labels): {len(labels)}')
    
    # get a scatter plot with thumbnail overlays
    get_scatter_plot_with_thumbnails(embeddings, filenames)


    example_images = [
    "P248_73665445941-C6_014_004_proj.tif",  # 0.82, Fe, 24
    "P278_73668090728-F5_007_002_proj.tif",  # 0, Fe, 0
    "P288_73669012104-E2_034_003_proj.tif",  # 1, X-ray, 48
    "P287_73668956345-E5_009_027_proj.tif",  # 0.1, X-ray, 48
    "P253_73666050044-C6_027_008_proj.tif",  # 0, Fe, 48
    ]
    
    # display example images for each cluster
    for i, example_image in enumerate(example_images):
        plot_nearest_neighbors_3x3(example_image, i, embeddings, filenames)

if __name__ == "__main__":
    main()
    # print(torch.cuda.device_count()) 
    # print(torch.cuda.get_device_name(0))