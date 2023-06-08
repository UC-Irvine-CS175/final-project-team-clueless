'''
Image Augmentations for use with the BPSMouseDataset class and PyTorch Transforms
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
from torchvision import transforms
import matplotlib.pyplot as plt
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
from PIL import Image
import os

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        # Normalizing uint numpy arrays representing images to a floating point
        # range between 0 and 1 brings the pixel values of the image to a common
        # scale that is compatible with most deep learning models.
        # Additionally, normalizing the pixel values can help to reduce the effects
        # of differences in illumination and contrast across different images, which
        # can be beneficial for model training. To normalize, we divide each pixel
        # value by the maximum value of the uint16 data type.

        # Normalize array values between 0 - 1
        img_array = img_array / np.iinfo(np.uint16).max

        # Conversion of uint16 -> float32
        img_normalized = img_array.astype(np.float32)

        # img_normalized = img_float / np.max(img_float)  # 65535.0

        return img_normalized

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.resize_width = resize_width
        self.resize_height = resize_height
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        img_resized = cv2.resize(img, (self.resize_width, self.resize_height))
        return img_resized

class ToThreeChannels(object):
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Convert the image to three channels

        args:
            img (np.ndarray): image to be converted to three channels.
        returns:
            torch.Tensor: image with three channels.
        """
        img_three_channels = np.repeat(img[..., np.newaxis], 3, -1)
        return img_three_channels
    
class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize


class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        img_flipped = cv2.flip(image, 0)  # vertically flip the image
        return img_flipped


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        img_flipped = cv2.flip(image, 1) # horizontally flip the image
        return img_flipped


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        if rotate in [90, 180, 270] :
            self.rotate = rotate
        else:
            self.rotate = 0

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotation (float):
                Optional parameter to specify a float for the number of degrees of rotation.
        Returns:
            np.ndarray
        '''
        s = image.shape
        cy = (s[0]-1)/2 # y center : float
        cx = (s[1]-1)/2 # x center : float
        M = cv2.getRotationMatrix2D((cx,cy),self.rotate,1) # rotation matrix
        return cv2.warpAffine(image,M,(s[1],s[0])) # Affine transformation to rotate the image and output size s[1],s[0]


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image):

        h, w = image.shape
        new_h, new_w = (self.output_height, self.output_width)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        img = image.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        #img_tensor = torch.from_numpy(image).unsqueeze(0)
        # image = image.transpose((2, 0, 1))
        return img_tensor
    
class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, resize, size):
        self.train_transform = transforms.Compose(
            [
                NormalizeBPS(),
                ResizeBPS(resize, resize),
                ### CHANGE THIS PART TO INCLUDE RANDOM AUGMENTATIONS
                transforms.RandomApply([HFlipBPS()], p=0.5),
                transforms.RandomApply([VFlipBPS()], p=0.5),
                transforms.RandomApply([RotateBPS(90)], p=0.5),
                RandomCropBPS(size, size),
                ####################################################
                ToThreeChannels(),
                ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                NormalizeBPS(),
                ResizeBPS(resize, resize),
                ToThreeChannels(),
                ToTensor()
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def main():
    """"""
    # load image using cv2
    path_to_data = root/'data'/'processed'
    img_key = 'P242_73665006707-A6_001_001_proj.tif'
    img_array = np.array(Image.open(os.path.join(path_to_data,img_key)))
    #img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    transforms = TransformsSimCLR(300, 254)
    img1, img2 = transforms(img_array)

    # torch transform returns a 3 x W x H image, we only show one color channel
    augmented_image_1 = img1.numpy()[0]
    augmented_image_2 = img2.numpy()[0]

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(img_array)
    axs[0].set_axis_off()
    axs[0].set_title("Original Image")

    axs[1].imshow(augmented_image_1)
    axs[1].set_axis_off()

    axs[2].imshow(augmented_image_2)
    axs[2].set_axis_off()
    plt.show()

    # test_augmentations_img = BPSAugmentations(img_array)

    # test_output_normalize= test_augmentations_img.normalize_bps()
    # test_output_resize = test_augmentations_img.resize_bps(500, 500)
    # test_output_vflip = test_augmentations_img.v_flip()
    # test_output_hflip = test_augmentations_img.h_flip()

    # #FIXME - Fix the rotate function to include superimage after
    # #test_output_rotate = test_augmentations_img.rotate(45)

    # # Output zoom increases or decreases the overall size
    # # of the image--must be followed up with resize in order to make
    # # sure all images are the same for PyTorch DataLoader
    # test_output_zoom = test_augmentations_img.zoom(3)
    # test_zoom_resize = BPSAugmentations(test_output_zoom).resize_bps(100,100)
    # print(test_output_zoom.shape)
    # print(test_zoom_resize.shape)





    # # Attempt to save

    # # Show as tensor
    # # plt.imshow(img_array[0][:][:])

    # # Show as np.array
    # plt.imshow(img_array)
    # plt.savefig('augmentations_b4_test.png')
    # # Show as np.array
    # plt.imshow(test_output_hflip)
    # plt.savefig('augmentations_hflip.png')
    # plt.imshow(test_output_vflip)
    # plt.savefig('augmentations_vflip.png')
    # plt.imshow(test_output_resize)
    # plt.savefig('augmentations_resize.png')
    
    # plt.imshow(test_zoom_resize)
    # plt.savefig('augmentations_after_zoom_resize.png')

if __name__ == "__main__":
    main()



### json file with a list of aug: object like structure -> image, : List of aougment preformed, and their description
### 



# Remove the black void:
# 1. Find dimensions to zoom in to
# 2. Call Zoom in function
# 3. Scale back to 64x64