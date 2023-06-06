import os
import random
import numpy as np
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.dataset.bps_dataset import BPSMouseDataset 
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)
from src.dataset.bps_datamodule import BPSDataModule

from src.vis_utils import(
    plot_gallery_from_1D,
    plot_2D_scatter_plot,
)

def preprocess_images(lt_datamodule: DataLoader) -> (np.ndarray, list):
    """
    The function flattens the 2-dimensional image into a 1-dimensional 
    representation required by dimensionality reduction algorithms (ie PCA). 
    
    When dealing with images, each pixel of the image corresponds to a feature. 
    In the original 2D image representation, the image is a matrix with rows and
    columns, where each entry represents the intensity or color value of a pixel. 

    Args:
        train_loader: A PyTorch DataLoader object containing the training dataset.
        num_images: The number of images to extract from the train_loader.

    Returns:
        X_flat: A numpy array of flattened images.
        all_labels: A list of labels corresponding to each flattened image.
    """
    # Initialize 2 empty lists and an image counter
    # 1 list to store the flattened images in a 1-dimensional representation
    # 1 list to store labels in a 1-dimensional representation

    images = []
    labels = []

    # Loop over each batch of images and labels by by enumerating over the train_loader
    for batch, (img, label) in enumerate(lt_datamodule()):
        # Set batch size to the number of images in the batch
        bs = len(img)

        for i in range(bs):
            # Apply a forward-chained squeeze to remove the batch and channels from the img
            temp_img = img[i].squeeze(0).squeeze(0)

            # Convert the image to a numpy array
            temp_img = temp_img.numpy()

            # Flatten the image using a built-in numpy function
            temp_img = temp_img.flatten()

            # To convert one hot encoded labels to a class index
            # Use np.argmax() function
            labels.append(np.argmax(label[i], axis=0))

            # Append the flattened image the list of all flattened images
            images.append(temp_img)
    

    # Convert the list of flattened images to a numpy array
    X_flat = np.array(images)

    # Return the np.ndarray of flattened images and the list of labels as a tuple
    return X_flat, labels

def perform_pca(X_flat: np.ndarray, n_components: int) -> tuple:
    """    
    PCA is commonly used for dimensionality reduction by projecting each data point onto only
    the first few principal components to obtain lower-dimensional data while preserving as
    much of the data's variation as possible.

    For more information: 
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args:
        X_flat: A numpy array of flattened images.
        n_components: The number of principal components to keep.

    Returns:
        pca: A PCA object that contains the principal components to be represented in the lower dimension.
        X_pca: A numpy array of the compressed image data with reduced dimensions.
    """
    # Initialize an PCA object from sklearn.decomposition and specify the number of components to keep
    pca = PCA(n_components = n_components)

    # Fit the PCA object to the flattened images. This step calculates the principal components
    # and learns the mean, components and variance-covariance matrix of X_flat.
    pca.fit(X_flat)

    # Apply dimensionality reduction to the flattened images which result in the compressed image
    # data with reduced dimensions.
    X_pca = pca.transform(X_flat)

    # Return the PCA object and the compressed image data which pca.transform() returns as a tuple in that order
    return pca, X_pca

def perform_tsne(X_reduced_dim: np.ndarray,
                 n_components: int,
                 lr: float = 150,
                 perplexity: int = 30,
                 angle: float = 0.2,
                 verbose: int = 2) -> np.ndarray:
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding) is an unsupervised non-linear dimensionality
    reduction technique for data exploration and visualizing high-dimensional data. Non-linear 
    dimensionality reduction means that the algorithm allows us to separate data that cannot be
    separated by a straight line.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Args:
        X_reduced_dim: A reduced dimensional representation of an original image.
        n_components: The number of components to calculate.
        lr: The learning rate for t-SNE.
        perplexity: The perplexity is related to the number of expected nearest neighbors.
        angle: The tradeoff between speed and accuracy for Barnes-Hut T-SNE.
        verbose: Verbosity level.
    """
    # Create a TSNE object from sklearn.manifold using the perform_tsne function parameters
    tsne = TSNE(n_components = n_components, learning_rate = lr, perplexity = perplexity, angle = angle, verbose = verbose)
    
    # Fit the TSNE object to X_reduced_dim
    tsne = tsne.fit_transform(X_reduced_dim)

    # Return the TSNE object
    return tsne

def create_tsne_cp_df(X_tsne: np.ndarray,
                      labels: list,
                      num_points: int) -> pd.DataFrame:
    """
    Create a dataframe that contains the lower dimensional t-SNE components and the labels for each image.

    Args:
        X_tsne: A numpy array of the lower dimensional t-SNE components.
        labels: A list of one hot encoded labels corresponding to each flattened image.
        num_points: The number of points to plot.

    Returns:
        cps_df: A dataframe that contains the lower dimensional t-SNE components and the labels for each image.
    """
    # Find the number of components in the X_tsne array by using the shape attribute
    # The array shape is (num_images, num_components)
    num_components = X_tsne.shape[1]
    # Create a pandas series from the labels list
    target_series = pd.Series(labels)
    # Create a list of column names for the dataframe by using a list comprehension and the num_components
    df_cols = ['CP' + str(i) for i in range(1, num_components + 1)]

    # Append the target column name to the list of column names
    targetName = 'target'
    df_cols.append(targetName)

    # Create a dataframe from the X_tsne array and the pandas series of labels with the column names restriced
    # to num_points number of rows
    cps_df = pd.DataFrame(columns=df_cols, data=np.column_stack((X_tsne[:num_points], target_series.iloc[:num_points])))

    # Convert the target column to an integer type
    cps_df[targetName] = cps_df[targetName].astype(int)

    # Create a dictionary that maps the integer target to the particle type
    int_to_particle = {
        0: 'Fe',
        1: 'X-ray'
    }

    # Create a new column called particle_type in the dataframe that contains the particle type as a string
    cps_df['particle_type'] = cps_df[targetName].map(int_to_particle)

    # Return the dataframe
    return cps_df

def main():
    """
    You may use this function to test your code.
    """
    # Fix the random seed to ensure reproducibility across runs
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_dir = root / 'data' / 'processed'
    train_meta_fname = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_dir = root / 'models' / 'baselines'
    batch_size = 2
    max_epochs = 3
    accelerator = 'auto'
    num_workers = 1
    acc_devices = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dm_stage = 'train'
    
    # Instantiate BPSDataModule 
    bps_datamodule = BPSDataModule(train_csv_file=train_meta_fname,
                                   train_dir=data_dir,
                                   val_csv_file=val_meta_fname,
                                   val_dir=data_dir,
                                   resize_dims=(64, 64),
                                   batch_size=batch_size,
                                   num_workers=num_workers)
    
    # Setup BPSDataModule which will instantiate the BPSMouseDataset objects
    # to be used for training and validation depending on the stage ('train' or 'val')
    bps_datamodule.setup(stage=dm_stage)

    image_stream_1d, all_labels = preprocess_images(lt_datamodule=bps_datamodule.train_dataloader)
    print(f'image_stream_1d.shape: {image_stream_1d.shape}')
    # Project the flattened images onto the principal components
    IMAGE_SHAPE = (64, 64)
    N_ROWS = 5
    N_COLS = 7
    N_COMPONENTS = N_ROWS * N_COLS

    # Perform PCA on the flattened images and specify the number of components to keep as 35
    pca, X_pca = perform_pca(X_flat=image_stream_1d, n_components=N_COMPONENTS)
    print(f'X_pca: {X_pca.shape}')

    # Plot the 1d array of flattened images
    plot_gallery_from_1D(title="Cell_Gallery_from_1D_Array",
                     images=image_stream_1d[:N_COMPONENTS],
                     n_row=N_ROWS,
                     n_col=N_COLS,
                     img_shape=IMAGE_SHAPE)
    print(f'X_pca.shape: {X_pca.shape}')

    # Plot the 1d array of flattened images after reducing the dimensionality using PCA
    plot_gallery_from_1D(title="PCA_Cell_Gallery_from_1D_Array",
                     images=pca.components_,
                     n_row=N_ROWS,
                     n_col=N_COLS,
                     img_shape=IMAGE_SHAPE)

    # Perform t-SNE on the flattened images before reducing the dimensionality using PCA
    X_tsne_direct = perform_tsne(X_reduced_dim=image_stream_1d, perplexity=30, n_components=2)
    print(f'X_tsne_direct.shape: {X_tsne_direct.shape}')
    # Perform t-SNE on the flattened images after reducing the dimensionality using PCA
    X_tsne_pca = perform_tsne(X_reduced_dim=X_pca, perplexity=30, n_components=2)
    print(f'X_tsne_pca.shape: {X_tsne_pca.shape}')
    tsne_df_direct = create_tsne_cp_df(X_tsne_direct, all_labels, 1000)
    print(tsne_df_direct.head())
    print(f'tsne_df_direct.shape: {tsne_df_direct.shape}')
    tsne_df_pca = create_tsne_cp_df(X_tsne_pca, all_labels, 1000)
    print(tsne_df_pca.head())
    print(f'tsne_df_pca.shape: {tsne_df_pca.shape}')
    plot_2D_scatter_plot(tsne_df_direct, 'tsne_direct_4hr_Gy_hi')
    plot_2D_scatter_plot(tsne_df_pca, 'tsne_pca_4hr_Gy_hi')

if __name__ == "__main__":
    main()