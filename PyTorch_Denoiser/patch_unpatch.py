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
import cv2

from utils import matlab_to_python_3d
from utils import python_to_matlab_4d


def create_patches_with_batches_and_resizing(input_data, patch_size, patch_stride, scales):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Resize image using 1 scale and create patches from resized image. Then repeat process for all
                 other scales.

    Input = (batches, channels, height, width)
    Output = (patches, channels, height, width)
    """

    print('')
    print('Creating Patches with create_patches_with_batches_and_resizing()...')
    print('Scales:', scales)

    print('')
    print('Before All Patches:', input_data.shape)              # torch.Size([105, 10, 230, 230])

    # -- Get dimensions
    channels = input_data.size()[1]
    height = input_data.size()[2]               # number of rows
    width = input_data.size()[3]                # number of cols
    # print('Channels:', channels)
    # print('Height (Num of Rows):', height)
    # print('Width (Num of Cols):', width)

    # -- Convert to Matlab format
    # print(input_data.shape)                           # torch.Size([105, 10, 230, 230])
    input_data = input_data.numpy()                     # Convert to numpy to use cv2.resize
    input_data = python_to_matlab_4d(input_data)        # Convert to height (num of rows) x width (num of cols) x channel x batch
    # print(input_data.shape)                           # (230, 230, 10, 105)

    # -- Iterate through scales
    for scale in scales:

        print('')
        print('Creating Patches with Scale', scale)

        scaled_dataset = []

        # -- Get new image size
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        # print(scaled_height)
        # print(scaled_width)

        # -- Iterate through batches
        for batch_num in range(input_data.shape[3]):
            # print(batch_num)

            # - Resize image
            scaled_image = cv2.resize(src=input_data[..., batch_num], dsize=(scaled_height, scaled_width), interpolation=cv2.INTER_CUBIC)
            # print(scaled_image.shape)                                 # (207, 207, 10) for scale=0.9

            # - Force it to have 3 dimensions here just in case channels = 1
            scaled_image = np.reshape(scaled_image, (scaled_height, scaled_width, channels))
            # print(scaled_image.shape)                                 # (207, 207, 10) for scale=0.9

            # - Convert back to Python format
            scaled_image = matlab_to_python_3d(scaled_image)
            # print(scaled_image.shape)                                 # (10, 207, 207) for scale=0.9

            scaled_dataset.append(scaled_image)

        scaled_dataset = np.asarray(scaled_dataset)
        # print(scaled_dataset.shape)                                   # (105, 10, 207, 207) for scale=0.9

        scaled_dataset = torch.from_numpy(scaled_dataset)

        # -- Create patches from resized images

        print('Before Patches: ', scaled_dataset.size())                # torch.Size([105, 10, 207, 207]) for scale=0.9

        # Reference: https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/16
        patches = scaled_dataset.unfold(1, channels, channels).unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)

        print('After Patches: ', patches.size())                        # torch.Size([105, 1, 5, 5, 10, 128, 128]) for scale=0.9

        flattened_patches = torch.flatten(patches, start_dim=0, end_dim=3)

        print('After Flattening Patches: ', flattened_patches.size())   # torch.Size([2625, 10, 128, 128]) for scale=0.9

        if scale == scales[0]:
            # If this is the first iteration in scales
            all_scaled_patches = flattened_patches
        else:
            all_scaled_patches = torch.cat((all_scaled_patches, flattened_patches), 0)

    print('')
    print('After All Patches:', all_scaled_patches.size())              # torch.Size([9870, 10, 128, 128])

    return all_scaled_patches
