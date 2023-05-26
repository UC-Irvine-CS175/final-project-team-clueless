""" Test for the BPSMouseDataset class."""

import unittest
import os
import torch
import pyprojroot
import sys
from torchvision import transforms, utils

sys.path.append(str(pyprojroot.here()))

from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.augmentation import NormalizeBPS, ResizeBPS, VFlipBPS, HFlipBPS, RotateBPS, RandomCropBPS, ToTensor
current_points_s3_test = 0
max_points_s3_test = 40
class TestBPSMouseDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = BPSMouseDataset('data.csv',
                                        os.path.join(pyprojroot.here(),'tests/test_dir'),
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

    def test_len(self):
        #self.assertEqual(len(self.dataset), 7092)
        self.assertEqual(len(self.dataset), 1, 'Length of dataset is inconsistent with number of files in csv file.')
        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")

    def test_getitem(self):
        img_tensor, particle_type = self.dataset[0]
        self.assertIsInstance(img_tensor, torch.Tensor, 'Image is not a tensor.')
        self.assertEqual(img_tensor.shape, (1, 200, 200), 'Image tensor is not the correct shape based on transforms.')
        self.assertIn(particle_type, ['X-ray', 'Fe'], 'Particle type is not in the list of possible particle types.')
        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 30
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")

    def tearDown(self):
        del self.dataset

if __name__ == '__main__':
    unittest.main()
