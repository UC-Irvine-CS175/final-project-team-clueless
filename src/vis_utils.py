import matplotlib.pyplot as plt
import torch
import typing as typing
import numpy as np
from torchvision import utils
import pandas as pd


# Faces Dataset Decomposition Tutorial https://github.com/olekscode/Examples-PCA-tSNE/blob/master/Python/Faces%20Dataset%20Decomposition.ipynb
def plot_gallery_from_2D(title, images, n_row, n_col, img_shape: tuple):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        # print("Testing for terminal hang")
        plt.imshow(comp, cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    # print("Testing for terminal hang pt. 2")    
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    # print("Testing for terminal hang pt. 3")   
    plt.savefig(f"{title}.png")

def plot_gallery_from_1D(title, images, n_row, n_col, img_shape: tuple):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        # print("Testing for terminal hang")
        plt.imshow(comp.reshape(img_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    # print("Testing for terminal hang pt. 2")    
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    # print("Testing for terminal hang pt. 3")   
    plt.savefig(f"{title}.png")

def show_image_and_label(image: torch.Tensor, label: str):
    """Show an image with a label"""
    plt.figure()
    image_np = image.permute(1, 2, 0).cpu().numpy()  # convert to numpy array and permute dimensions
    plt.imshow(image_np)
    plt.title(f"Label: {label}")
    
def plot_dataloader_batch(img_batch, label_batch):
    images_batch = img_batch
    plt.figure()
    grid_im = utils.make_grid(images_batch)
    #plt.imshow(grid_im.numpy().squeeze())
    plt.imshow(grid_im.numpy().transpose((1, 2, 0)))
    plt.title(f"Label: {label_batch}")
    plt.savefig('batch_sample.png')

def show_label_batch(images_batch: torch.Tensor, label_batch: str):
    """Show image with label for a batch of samples."""
    plt.figure()
    # batch_size = len(images_batch)
    # im_size = images_batch.size(2)
    # grid_border_size = 2

    # grid is a 4 dimensional tensor (channels, height, width, number of images/batch)
    # images are 3 dimensional tensor (channels, height, width), where channels is 1
    # utils.make_grid() takes a 4 dimensional tensor as input and returns a 3 dimensional tensor
    # the returned tensor has the dimensions (channels, height, width), where channels is 3
    # the returned tensor represents a grid of images
    grid = utils.make_grid(images_batch, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.title(f"Label: {label_batch}")
    plt.savefig('batch_sample.png')

def plot_2D_scatter_plot(cps_df: pd.DataFrame,
                         scatter_fname: str) -> None:
    """
    Create a 2D scatter plot of the t-SNE components when the number of components is 2.
    https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3

    Args:
        cps_df: A dataframe that contains the lower dimensional t-SNE components and the labels for each image.
    
    Returns:
        None
    """
    # Create a figure
    plt.figure()
    # Create a scatter plot of the t-SNE components using the first two components and the target column
    # which will be assigned a different color to each data point based on its value
    plt.scatter(cps_df.CP1, cps_df.CP2, c=cps_df.target)
    # Get the unique targets from the dataframe
    unique_targets = cps_df.target.unique()
    # Generate a list of colors using the rainbow colormap from matplotlib with the number of colors
    # equal to the number of unique targets
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
    # iterate over the unique targets and colors by using the zip function
    for target, color in zip(unique_targets, colors):
        # Create a boolean mask for the target by comparing the target column to the target
        indices = cps_df['target'] == target
        # Create a scatter plot using the boolean mask and the first two components
        plt.scatter(cps_df.loc[indices, 'CP1'], cps_df.loc[indices, 'CP2'], color=color, label=target)
    # Add a legend to the plot
    plt.legend(title='Particle Type')
    # Add axis labels and a title to the plot
    plt.xlabel('CP1')
    plt.ylabel('CP2')
    plt.title(f'{scatter_fname} 2 Component t-SNE')
    plt.savefig(f"{scatter_fname}.png")

def plot_3D_scatter_plot(cps_df: pd.DataFrame,
                         scatter_fname: str) -> None:
    """
    Create a 2D scatter plot of the t-SNE components when the number of components is 2.
    https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3

    Args:
        cps_df: A dataframe that contains the lower dimensional t-SNE components and the labels for each image.
    
    Returns:
        None
    """
    # Define a color map
    cmap = plt.cm.rainbow
    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create a scatter plot of the t-SNE components using the first two components and the target column
    # which will be assigned a different color to each data point based on its value
    scatter = ax.scatter(cps_df['CP1'], cps_df['CP2'], cps_df['CP3'], c=cps_df['target'], cmap=cmap)
    cbar = plt.colorbar(scatter)

    # Set labels and title
    ax.set_xlabel('CP1')
    ax.set_ylabel('CP2')
    ax.set_zlabel('CP3')
    ax.set_title(f'{scatter_fname} 3 Component t-SNE')

    # Show the plot
    plt.savefig(f"{scatter_fname}.png")
    plt.show()

