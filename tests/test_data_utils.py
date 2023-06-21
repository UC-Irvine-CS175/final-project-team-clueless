import os
import unittest
import csv
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from io import BytesIO
import pyprojroot
import sys

# root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
# test_data_dir = root/ 'tests' / 'test_dir'
# print(test_data_dir)

sys.path.append(str(pyprojroot.here()))

root = pyprojroot.here()
test_data_dir = root/ 'tests' / 'test_dir'

from src.data_utils import (
    get_bytesio_from_s3,
    get_file_from_s3,
    save_tiffs_local_from_s3,
    export_subset_meta_dose_hr,
    train_test_split_subset_meta_dose_hr,
)
current_points_s3_test = 0
max_points_s3_test = 50
class TestS3Functions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        cls.bucket_name = 'nasa-bps-training-data'
        cls.s3_file_path = "Microscopy/train"
        cls.file_name = 'meta.csv'
        cls.local_dir = test_data_dir
        # cls.local_file_path = os.path.join(os.path.join(sys.path[0], cls.local_dir), cls.file_name)
        cls.local_file_path = os.path.join(cls.local_dir, cls.file_name)
        cls.s3_full_fpath = os.path.join(cls.s3_file_path, cls.file_name)
        cls.s3_full_fpath = cls.s3_full_fpath.replace(os.sep, '/') # new to fix on Windows machines
        cls.subset_csv_path_Gy_hi = None
        cls.subset_csv_path_Gy_med = None
        cls.subset_csv_path_Gy_low = None

        # Create local directory for testing
        os.makedirs(cls.local_dir, exist_ok=True)

        # Make dummy csv file since save_tiff_from_s3 requires a csv file to fetch all
        # the tif files from s3. This dummy file only has one row.
        # Define the data as a list of tuples
        data = [('P242_73665006707-A6_003_013_proj.tif', 0.82, 'Fe', 4)]
        # Define the headers of the CSV file
        headers = ['filename', 'dose_Gy', 'particle_type', 'hr_post_exposure']
        cls.dummy_csv_file = f'{test_data_dir}/data.csv'

        # Create the directory for the dummy csv file
        # os.makedirs(os.path.join(sys.path[0],'test_dir'), exist_ok=True)
        os.makedirs(test_data_dir, exist_ok=True)

        # Open a new CSV file in write mode
        with open(cls.dummy_csv_file, mode='w', newline='') as file:

            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the headers to the CSV file
            writer.writerow(headers)

            # Write the data to the CSV file
            writer.writerows(data)

    @classmethod
    def tearDownClass(cls):
        # FIXME: Every other run fails because paths are not found in test_export_subset_meta_dose_hr
        # & test_train_test_split_subset_meta_dose_hr
        # Clean up local directory for testing
        # os.remove(cls.local_file_path)
        # os.rmdir(cls.local_dir)
        # Clean up test_dir directory
        
        # os.remove(os.path.join(sys.path[0], 'test_dir/meta_dose_hi_hr_4_post_exposure.csv'))
        # os.remove(os.path.join(sys.path[0], 'test_dir/meta_dose_med_hr_4_post_exposure.csv'))
        # os.remove(os.path.join(sys.path[0], 'test_dir/meta_dose_low_hr_4_post_exposure.csv'))
        # os.remove('test_dir/meta_dose_hi_hr_4_post_exposure_train.csv')
        # os.remove('test_dir/meta_dose_hi_hr_4_post_exposure_test.csv')
        # os.remove('test_dir/' + 'P242_73665006707-A6_003_013_proj.tif')
        # os.remove(os.path.join(sys.path[0], cls.dummy_csv_file))
        # os.remove(os.path.join(sys.path[0], cls.local_file_path))
        # os.rmdir(cls.local_dir)
            
        pass

    def test_get_bytesio_from_s3(self):
        # actual_data is a BytesIO object
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.s3_full_fpath)
        file_contents = response["Body"].read()
        actual_data = get_bytesio_from_s3(self.s3_client, self.bucket_name, self.s3_full_fpath)
        self.assertIsInstance(actual_data, BytesIO, "The data returned is not a BytesIO object")
        self.assertEqual(actual_data.getvalue(), file_contents, "The data in the BytesIO object is not the same as the data in the s3 file")
        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")
            

    def test_get_file_from_s3(self):
        expected_local_file_path = self.local_file_path

        actual_file_path = get_file_from_s3(self.s3_client,
                                            self.bucket_name,
                                            self.s3_full_fpath,
                                            self.local_dir)
        with open(expected_local_file_path, "r") as f:
            expected_data = f.read()
        with open(actual_file_path, "r") as f:
            actual_data = f.read()
        self.assertEqual(actual_data, expected_data, "The data in the local file is not the same as the data in the s3 file")
        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")
        # os.remove(self.local_file_path)
        # os.rmdir('test_dir')

    def test_save_tiff_from_s3(self):
        # Create the path to the local tif file using the data variable
        local_tif_path = test_data_dir

        save_tiffs_local_from_s3(self.s3_client,
                            self.bucket_name,
                            self.s3_file_path,
                            self.dummy_csv_file,
                            local_tif_path)
        assert os.path.exists(local_tif_path)

        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")

    def test_export_subset_meta_dose_hr(self):
        print(self.local_file_path)
        out_csv_path, subset_size_Gy_hi = export_subset_meta_dose_hr(dose_Gy_specifier='hi',
                                                                hr_post_exposure_val=4,
                                                                in_csv_path_local=self.local_file_path,
                                                                out_dir_csv=self.local_dir)
        self.subset_csv_path_Gy_hi = out_csv_path

        out_csv_path, subset_size_Gy_med = export_subset_meta_dose_hr(dose_Gy_specifier='med',
                                                                hr_post_exposure_val=4,
                                                                in_csv_path_local=self.local_file_path,
                                                                out_dir_csv=self.local_dir)
        self.subset_csv_path_Gy_med = out_csv_path

        out_csv_path, subset_size_Gy_low = export_subset_meta_dose_hr(dose_Gy_specifier='low',
                                                                hr_post_exposure_val=4,
                                                                in_csv_path_local=self.local_file_path,
                                                                out_dir_csv=self.local_dir)   
        self.subset_csv_path_Gy_low = out_csv_path

        # Assert when Gy is hi, hr is 4, that the subset size is 8866
        self.assertEqual(subset_size_Gy_hi, 8866, "Selection of dose_Gy = hi and hr_post_exposure = 4 failed.")

        # Assert when Gy is med, hr is 4, that the subset size is 9308
        self.assertEqual(subset_size_Gy_med, 9308, "Selection of dose_Gy = med and hr_post_exposure = 4 failed.")
        
        #Assert when Gy is lo, hr is 4, that the subset size is 9435
        self.assertEqual(subset_size_Gy_low, 9435, "Selection of dose_Gy = low and hr_post_exposure = 4 failed.")

        self.assertIsNotNone(self.subset_csv_path_Gy_hi, "The path to the subset csv file for dose_Gy = hi and hr_post_exposure = 4 is None")

        self.assertIsNotNone(self.subset_csv_path_Gy_med, "The path to the subset csv file for dose_Gy = med and hr_post_exposure = 4 is None")

        self.assertIsNotNone(self.subset_csv_path_Gy_low, "The path to the subset csv file for dose_Gy = low and hr_post_exposure = 4 is None")

        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")

    def test_train_test_split(self):
        # Call the function, assign the paths, get the num_lines in csv file,
        #  then assertGreater(train > test)
        # csv_path_Gy_hi = os.path.join(sys.path[0], 'test_dir/meta_dose_hi_hr_4_post_exposure.csv')   # 'test_dir/meta_dose_hi_hr_4_post_exposure.csv'
        csv_path_Gy_hi = os.path.join(test_data_dir, 'meta_dose_hi_hr_4_post_exposure.csv')   # 'test_dir/meta_dose_hi_hr_4_post_exposure.csv'
        
        train_csv_local_path, test_csv_local_path = \
            train_test_split_subset_meta_dose_hr(csv_path_Gy_hi,
                                                 0.2,
                                                 self.local_dir,
                                                 42,
                                                 'particle_type')
        df_train = pd.read_csv(train_csv_local_path)
        df_test = pd.read_csv(test_csv_local_path)

        self.assertGreater(df_train.shape[0], df_test.shape[0], "The training set is smaller than the test set")

        if self._outcome.success:
            global current_points_s3_test
            current_points_s3_test += 10
        print(f"\n\nCurrent points: {current_points_s3_test}/{max_points_s3_test}")

# def main():

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)