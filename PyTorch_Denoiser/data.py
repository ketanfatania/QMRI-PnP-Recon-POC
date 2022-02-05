"""
==============================================================================================================
Author: Ketan Fatania
Affiliation: University of Bath
Email: kf432@bath.ac.uk
GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
Date: --/09/2020 to --/02/2022

@inproceedings{ref:fatania2022,
    author = {Fatania, Ketan and Pirkl, Carolin M. and Menzel, Marion I. and Hall, Peter and Golbabaee, Mohammad},
    booktitle = {2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
    title = {A Plug-and-Play Approach to Multiparametric Quantitative MRI: Image Reconstruction using Pre-Trained Deep Denoisers},
    code = {https://github.com/ketanfatania/QMRI-PnP-Recon-POC},
    year = {2022}
    }
==============================================================================================================
"""

import numpy as np
import torch

from data_utils import load_real_fisp_float64
from patch_unpatch import create_patches_with_batches_and_resizing
from utils import apply_data_aug_for_patches
from utils import resize_images_with_batches            # Uses cv2
from utils import generate_noisy_data_fly_np
from utils import generate_noise_map
import scale_rescale
from scale_rescale import scale_image_with_batch


class TSMIDataset(torch.utils.data.Dataset):

    def __init__(self, mode='train', patches_mode='off',
                 train_settings=None, test_settings=None, device='cpu'):

        self.mode = mode
        self.patches_mode = patches_mode
        self.train_settings = train_settings
        self.test_settings = test_settings
        self.device = device

        # -- Get path of training and testing data
        if mode == 'train':
            path = self.train_settings.training_data_path
            dataset = self.train_settings.dataset
            dataset_float32or64_mode = self.train_settings.dataset_float32or64_mode
        elif mode == 'test':
            path = self.test_settings.testing_data_path
            dataset = self.test_settings.dataset
            dataset_float32or64_mode = self.test_settings.dataset_float32or64_mode

        # -- Load chosen data using correct method
        if 'real_fisp' in dataset:
            print('Loading data using load_real_fisp_float64 function...')
            x_ground_truth = load_real_fisp_float64(path, sort=False)
            print('Shape of data loaded: ' + str(x_ground_truth.shape))
        else:
            print('Select a valid dataset loading method')
            exit()

        # -- Choose how much of the data is loaded
        if mode == 'train':

            print('Using all training data loaded ...')

        elif mode == 'test':

            print('Using slice ' + str(self.test_settings.slice_num+1) + ' from a total of ' + str(x_ground_truth.shape[0]) + ' loaded slices ...')

            x_ground_truth = x_ground_truth[self.test_settings.slice_num:self.test_settings.slice_num+1]
            # print(x_ground_truth.shape)               # (1, 10, 230, 230)

        # -- Enforce float32 or float64 data
        # Note: fisp data is float64
        print('')
        print('Before enforcing data dtype:', x_ground_truth.dtype)
        if dataset_float32or64_mode == '32':
            print('Converting data dtype to Float32...')
            x_ground_truth = np.float32(x_ground_truth)
        elif dataset_float32or64_mode == '64':
            print('Converting data dtype to Float64...')
            x_ground_truth = np.float64(x_ground_truth)
        else:
            print('Select a valid float32or64_mode')
            exit()
        print('After enforcing data dtype:', x_ground_truth.dtype)

        # -- Initialise noisy input variable
        x_noisy_input = x_ground_truth

        # -- Create training and testing dataset
        if mode == 'train':
            print('')
            if patches_mode == 'off':
                print('patches_mode = off')
                print('Patches mode off was not used in ISBI 2022 paper')
                exit()

            elif patches_mode == 'on':
                print('patches_mode = on')

                if self.train_settings.patches_on_mode == 'ISBI_2022_Paper':
                    # -- For DRUNet Single-Noise Denoiser and Multi-Noise Denoiser
                    # - Order: resize data, create patches, apply augmentations and then normalize 0to1
                    print('')
                    print('Creating dataset for DRUNet Single-Noise Denoiser/Multi-Noise Denoiser...')

                    normalized_augmented_patches = create_patches_with_batches_and_resizing(torch.from_numpy(x_noisy_input),
                                                                                            patch_size=self.train_settings.patch_size, patch_stride=self.train_settings.patch_stride,
                                                                                            scales=self.train_settings.resize_scales
                                                                                            ).numpy()
                    normalized_augmented_patches = apply_data_aug_for_patches(normalized_augmented_patches)
                    normalized_augmented_patches = scale_image_with_batch(normalized_augmented_patches, lower_bounds=0, upper_bounds=1, debug_mode='off')
                    # print(normalized_augmented_patches.shape)           # (9870, 10, 128, 128) for patch size 128, stride 17 and scales scales=[1, 0.9, 0.8, 0.7]

                    self.x_noisy_input_train = torch.from_numpy(normalized_augmented_patches)
                    self.x_ground_truth_train = torch.from_numpy(normalized_augmented_patches)

                else:
                    print('')
                    print('Select a Method of Data Creation for TrainingMode - PatchesOn')
                    exit()

            # -- Send all training data to GPU or not
            print('')
            if train_settings.load_all_data_to_gpu == 'on':
                print('Loading all training data to device...')
                self.x_noisy_input_train = self.x_noisy_input_train.to(device)
                self.x_ground_truth_train = self.x_ground_truth_train.to(device)
            elif train_settings.load_all_data_to_gpu == 'off':
                print('All training data not loaded to device...')

        elif mode == 'test':
            print('Creating test data ...')

            if patches_mode == 'off':
                print('Creating test data with patches mode off ...')

                if self.test_settings.patches_off_mode == 'DRUNet_Single_Level':
                    # -- For DRUNet Single-Noise Denoiser (10ch)
                    # - Order: crop to 224x224, normalize 0to1, generate noisy data
                    print('')
                    print('Creating dataset for DRUNet_Single_Level (10ch) ...')

                    x_noisy_input = resize_images_with_batches(dataset=x_noisy_input, height=test_settings.num_rows, width=test_settings.num_cols)
                    x_ground_truth = resize_images_with_batches(dataset=x_ground_truth, height=test_settings.num_rows, width=test_settings.num_cols)

                    normalized_images = scale_image_with_batch(x_noisy_input, lower_bounds=0, upper_bounds=1,
                                                               debug_mode='on')
                    global global_scale_image_batch_x_min_values_noisy        # Comment out if this line is elsewhere in data.py
                    global global_scale_image_batch_x_max_values_noisy        # Comment out if this line is elsewhere in data.py
                    global_scale_image_batch_x_min_values_noisy = scale_rescale.global_scale_image_batch_x_min_values
                    global_scale_image_batch_x_max_values_noisy = scale_rescale.global_scale_image_batch_x_max_values

                    x_noisy_input = generate_noisy_data_fly_np(normalized_images, settings=test_settings)

                    self.x_noisy_input_test = torch.from_numpy(x_noisy_input)
                    self.x_ground_truth_test = torch.from_numpy(x_ground_truth)

                elif self.test_settings.patches_off_mode == 'DRUNet_Multi_Level':
                    # -- For DRUNet Single-Noise Denoiser (11ch)
                    # - Order: crop to 224x224, normalize 0to1, generate noisy data with noise map
                    print('')
                    print('Creating dataset for DRUNet_Multi_Level (11ch) ...')

                    x_noisy_input = resize_images_with_batches(dataset=x_noisy_input, height=test_settings.num_rows, width=test_settings.num_cols)
                    x_ground_truth = resize_images_with_batches(dataset=x_ground_truth, height=test_settings.num_rows, width=test_settings.num_cols)

                    normalized_images = scale_image_with_batch(x_noisy_input, lower_bounds=0, upper_bounds=1,
                                                               debug_mode='on')
                    # global global_scale_image_batch_x_min_values_noisy        # Comment out if this line is elsewhere in data.py
                    # global global_scale_image_batch_x_max_values_noisy        # Comment out if this line is elsewhere in data.py
                    global_scale_image_batch_x_min_values_noisy = scale_rescale.global_scale_image_batch_x_min_values
                    global_scale_image_batch_x_max_values_noisy = scale_rescale.global_scale_image_batch_x_max_values

                    noise_map = generate_noise_map(input_data=normalized_images, settings=test_settings)
                    x_noisy_input = generate_noisy_data_fly_np(normalized_images, settings=test_settings)
                    x_noisy_input = np.concatenate((x_noisy_input, noise_map), axis=1)

                    self.x_noisy_input_test = torch.from_numpy(x_noisy_input)
                    self.x_ground_truth_test = torch.from_numpy(x_ground_truth)

                else:
                    print('')
                    print('Select a Method of Data Creation for TestingMode - PatchesOff')
                    exit()

            elif patches_mode == 'on':
                print('Creating test data with patches mode on ...')
                print('Patches mode on was not used in ISBI 2022 paper')
                exit()

    def __len__(self):
        if self.mode == 'train':
            return len(self.x_ground_truth_train)
        elif self.mode == 'test':
            return len(self.x_ground_truth_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.x_noisy_input_train[index], self.x_ground_truth_train[index]
        elif self.mode == 'test':
            return self.x_noisy_input_test[index], self.x_ground_truth_test[index]
