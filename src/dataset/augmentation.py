'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision
from torchvision import transforms, utils
from PIL import Image

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        
        if np.max(img_array)-np.min(img_array) == 0:
            return img_array

        return (img_array-np.min(img_array))/(np.max(img_array)-np.min(img_array))

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.resize_height = resize_height
        self.resize_width = resize_width
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        # Convert numpy image to PIL image
        img = Image.fromarray(img).convert('L')

        # Resize the image.
        transform = torchvision.transforms.Resize((self.resize_height, self.resize_width))
        img = transform(img)

        # Convert PIL image to numpy image
        img = np.array(img)
        """
        confusion here - it says return a torch.Tensor but the code returns np.array and mine too.
        my code:
        a = np.resize(img, (self.resize_height, self.resize_width))
        return a
        """
        return img
    

class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        return np.flipud(image)


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        return np.fliplr(image)


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        assert rotate in [90, 180, 270], 'rotate must be either 90, 180 or 270 degrees'
        self.rotate = rotate

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        return np.rot90(image, k=self.rotate // 90)


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
        height, width = image.shape[:2]
        new_h = self.output_height
        new_w = self.output_width

        top = np.random.randint(0, height - new_h)
        left = np.random.randint(0, width - new_w)

        img = image[top: top + new_h, left: left + new_w]

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # numpy image: H x W x C
        # torch image: C x H x W

        img = Image.fromarray(image)

        return transforms.ToTensor()(img)

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

def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'C:\\Users\\sammy\\Desktop\\Projects\\compsci175\\download-dataset-augmentations-dataloader-sammypham\Microscopy\\train\\P242_73665006707-A6_001_001_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    print(test_resize.shape)
    type(test_resize)

if __name__ == "__main__":
    main()