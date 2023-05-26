import unittest
import cv2
import numpy as np
import torch
import torchvision
from typing import Any, Tuple
import pyprojroot
import sys

sys.path.append(str(pyprojroot.here()))
#print(f'Path: {sys.path[0]}')
#print(f'Path: {pyprojroot.here()}')

from src.dataset.augmentation import NormalizeBPS, ResizeBPS, ZoomBPS, VFlipBPS, HFlipBPS, RotateBPS, RandomCropBPS, ToTensor

current_points_test = 0
max_points_test = 80
class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Create sample image
        self.img = np.random.rand(256, 256).astype(np.uint16)

        # Define augmentations
        self.normalize = NormalizeBPS()
        self.resize = ResizeBPS(resize_height=128, resize_width=128)
        self.zoom = ZoomBPS(zoom=2)
        self.vflip = VFlipBPS()
        self.hflip = HFlipBPS()
        self.rotate = RotateBPS(rotate=90)
        self.randomcrop = RandomCropBPS(output_height=128, output_width=128)
        self.to_tensor = ToTensor()

    def tearDown(self):
        pass

    def test_normalize(self):
        # Test normalize function
        img_normalized = self.normalize(self.img)
        self.assertTrue(isinstance(img_normalized, np.ndarray), 'Image is not a numpy array.')
        self.assertLessEqual(np.max(img_normalized), 1.0, 'Image is not normalized.')

        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_resize(self):
        # Test resize function
        img_resized = self.resize(self.img)
        self.assertTrue(isinstance(img_resized, np.ndarray), 'Image is not a numpy array.')
        self.assertEqual(img_resized.shape, (128, 128), 'Image is not the correct shape based on resize parameters.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_zoom(self):
        # Test zoom function
        img_zoomed = self.zoom(self.img)
        self.assertTrue(isinstance(img_zoomed, np.ndarray), 'Image is not a numpy array.')
        self.assertEqual(img_zoomed.shape, (512, 512), 'Image is not the correct shape based on zoom parameters.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_vflip(self):
        # Test vertical flip function
        img_flipped = self.vflip(self.img)
        self.assertTrue(isinstance(img_flipped, np.ndarray))
        self.assertTrue(np.allclose(img_flipped, np.flipud(self.img)), 'Image is not flipped vertically.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_hflip(self):
        # Test horizontal flip function
        img_flipped = self.hflip(self.img)
        self.assertTrue(isinstance(img_flipped, np.ndarray))
        self.assertTrue(np.allclose(img_flipped, np.fliplr(self.img)), 'Image is not flipped horizontally.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_rotate(self):
        # Test rotation function
        img_rotated = self.rotate(self.img)
        self.assertTrue(isinstance(img_rotated, np.ndarray))
        self.assertEqual(img_rotated.shape, (256, 256))
        self.assertTrue(np.allclose(img_rotated, np.rot90(self.img)), 'Image is not rotated 90 degrees.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_randomcrop(self):
        # Test random crop function
        img_cropped = self.randomcrop(self.img)
        self.assertTrue(isinstance(img_cropped, np.ndarray))
        self.assertEqual(img_cropped.shape, (128, 128), 'Image is not the correct shape based on crop parameters.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

    def test_to_tensor(self):
        # Test to tensor function
        img_tensor = self.to_tensor(self.img.astype(np.float32))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))
        self.assertEqual(img_tensor.shape, (1, 256, 256), 'Image is not the correct shape based on tensor parameters.')
        if self._outcome.success:
            global current_points_test
            current_points_test += 10
        print(f"\n\nCurrent points: {current_points_test}/{max_points_test}")

if __name__ == '__main__':
    unittest.main()