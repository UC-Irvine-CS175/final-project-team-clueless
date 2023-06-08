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

from src.dataset.simclr_dataloader.bps_dataset import BPSMouseDataset
from src.dataset.simclr_dataloader.augmentation import TransformsSimCLR
from src.dataset.simclr_dataloader.bps_datamodule import BPSDataModule

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
    data_dir:           str = os.path.join(root,'data','raw')
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_vis_dir:       str = os.path.join(root, 'visualizations', 'simclr_knn')
    save_models_dir:    str = os.path.join(root, 'models', 'simclr')
    batch_size:         int = 8 # 128: change once able to train successfully
    max_epochs:         int = 1
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
    
    
    wandb.init(project="SimCLR",
               dir=config.save_vis_dir,
               config=
               {
                   "architecture": "SimCLR",
                   "dataset": "BPS Microscopy"
               })
    model = SimCLR()
    trainer = pl.Trainer(max_epochs=config.max_epochs, devices=1, accelerator=config.accelerator)
    trainer.fit(model, bps_datamodule.train_dataloader())
    model.eval()
    embeddings, filenames, labels = generate_embeddings(model, bps_datamodule.val_dataloader())
    print(f'embeddings.shape: {embeddings.shape}')
    print(f'len(filenames): {len(filenames)}')
    print(f'len(labels): {len(labels)}')
    plot_knn_examples(embeddings, filenames, labels)
    wandb.finish()

    # You could use the pretrained model and train a classifier on top.
    pretrained_resnet_backbone = model.backbone
    
    # you can also store the backbone and use it in another code
    state_dict = {"resnet18_parameters": pretrained_resnet_backbone.state_dict()}
    torch.save(state_dict, os.path.join(config.save_models_dir, "model.pth"))


if __name__ == "__main__":
    main()