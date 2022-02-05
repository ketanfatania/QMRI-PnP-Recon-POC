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

import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def show_image(image):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Display an image
    """

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar(orientation='horizontal')
    plt.show()


def compare_images(gt_image, noisy_image, test_image, clim):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Compare three images in one figure
    """

    plt.figure()

    ax = plt.subplot(1, 3, 1)
    plt.tight_layout()
    ax.axis('off')
    plt.imshow(gt_image, cmap='gray')
    plt.clim([gt_image.min(), gt_image.max()]) if clim == 'gt' else plt.clim(clim)
    plt.title('Ground Truth')
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(1, 3, 2)
    plt.tight_layout()
    ax.axis('off')
    plt.imshow(noisy_image, cmap='gray')
    plt.clim([gt_image.min(), gt_image.max()]) if clim == 'gt' else plt.clim(clim)
    plt.title('Noisy Image')
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(1, 3, 3)
    plt.tight_layout()
    ax.axis('off')
    plt.imshow(test_image, cmap='gray')
    plt.clim([gt_image.min(), gt_image.max()]) if clim == 'gt' else plt.clim(clim)
    plt.title('Output from CNN')
    plt.colorbar(orientation='horizontal')

    plt.show()


def generate_noisy_data_fly(x_gt, settings):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: To add noise to Torch Data, Only used in main.py
    """

    noise = np.random.normal(settings.noise_gauss_mean, settings.noise_gauss_std, x_gt.shape)

    if settings.noise_gauss_dtype == '32':
        # print('Using Float32 Noise...')
        noise = np.float32(noise)
    elif settings.noise_gauss_dtype == '64':
        # print('Using Float64 Noise...')
        noise = np.float64(noise)               # Don't need to convert to float64 as it already is by default - here for completeness
    # print(noise.dtype)                        # float32

    noise = torch.from_numpy(noise).to(settings.device)             # Convert noise to torch type and to device
    # print(noise.dtype)                                            # torch.float32

    x_noisy_inputs = x_gt + noise
    # print(x_noisy_inputs.dtype)               # torch.float32 if settings.noise_gauss_dtype='32' and torch.float64 if settings.noise_gauss_dtype='64'

    return x_noisy_inputs


def generate_noisy_data_fly_np(x_gt, settings):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: To add noise to Numpy Data, Only used in data.py
    """

    noise = np.random.normal(settings.noise_gauss_mean, settings.noise_gauss_std, x_gt.shape)
    # print(noise.dtype)                            # float64

    if settings.noise_gauss_dtype == '32':
        # print('Using Float32 Noise...')
        noise = np.float32(noise)
    elif settings.noise_gauss_dtype == '64':
        # print('Using Float64 Noise...')
        noise = np.float64(noise)                   # Don't need to convert to float64 as it already is by default - here for completeness
    # print(noise.dtype)

    x_noisy_inputs = x_gt + noise
    # print(x_noisy_inputs.dtype)                   # float32 if settings.noise_gauss_dtype='32' and float64 if settings.noise_gauss_dtype='64'

    return x_noisy_inputs


def generate_noisy_data_fly_blind(x_gt, settings):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Use with generate_noise_map_fly(), To add noise to Torch Data, Only used in main.py
    """

    noise = np.random.normal(settings.noise_gauss_blind_mean_tmp_val, settings.noise_gauss_blind_std_tmp_val, x_gt.shape)

    if settings.noise_gauss_blind_dtype == '32':
        # print('Using Float32 Noise...')
        noise = np.float32(noise)
    elif settings.noise_gauss_blind_dtype == '64':
        # print('Using Float64 Noise...')
        noise = np.float64(noise)               # Don't need to convert to float64 as it already is by default - here for completeness
    # print(noise.dtype)                        # float32

    noise = torch.from_numpy(noise).to(settings.device)             # Convert noise to torch type and to device
    # print(noise.dtype)                                            # torch.float32

    x_noisy_inputs = x_gt + noise
    # print(x_noisy_inputs.dtype)               # torch.float32 if settings.noise_gauss_dtype='32' and torch.float64 if settings.noise_gauss_dtype='64'

    return x_noisy_inputs


def generate_noise_map_fly(input_data, settings):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Use with generate_noisy_data_fly_blind(), Should be able to use it for both torch and numpy data,
                 Only used in main.py, Used for training
    """

    # -- Initialize variables
    batch_size, channels, height, width = input_data.shape
    # print(batch_size, channels, height, width)                    # (16 1 128 128)
    channels = 1                                                    # Noise map only needs to be one channel

    # -- Generate single sigma value
    noise_sigma_orig = np.random.uniform(low=settings.noise_gauss_blind_std[0], high=settings.noise_gauss_blind_std[1])

    # -- Change sigma value to correct dtype
    if settings.noise_gauss_blind_dtype == '32':
        # print('Using Float32 Noise...')
        noise_sigma = np.float32(noise_sigma_orig)
    elif settings.noise_gauss_blind_dtype == '64':
        # print('Using Float64 Noise...')
        noise_sigma = np.float64(noise_sigma_orig)      # Don't need to convert to float64 as it already is by default - here for completeness
    # print(noise_sigma.dtype)                          # float32

    # -- Generate noise map
    noise_map = np.tile(noise_sigma, (batch_size, channels, height, width))
    # print(noise_map)                                # Every element has the same value
    # print(noise_map.shape)                          # (16, 1, 128, 128)

    return noise_sigma_orig, noise_map


def generate_noise_map(input_data, settings):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Should be able to use it for both torch and numpy data, Only used in data.py, Used for testing
    """

    batch_size, channels, height, width = input_data.shape
    # print(batch_size, channels, height, width)                    # (1, 10, 224, 224)
    channels = 1                                                    # Noise map only needs to be one channel

    # - Generate noise map
    noise_map = np.tile(settings.noise_gauss_std, (batch_size, channels, height, width))
    # print(noise_map)                                # Every element has the same value
    # print(noise_map.shape)                          # (1, 1, 224, 224)

    if settings.noise_gauss_dtype == '32':
        print('Using Float32 Noise Map...')
        noise_map = np.float32(noise_map)
    elif settings.noise_gauss_dtype == '64':
        print('Using Float64 Noise Map...')
        noise_map = np.float64(noise_map)               # Don't need to convert to float64 as it already is by default - here for completeness
    # print(noise.dtype)

    return noise_map


def sao_yan_data_augmentation(image, mode):

    """
    Author: SaoYan
    Straight copy from SaoYan's PyTorch implementation of Zhang et al's Matlab DnCNN code
    SaoYan's GitHub Repo: https://github.com/SaoYan/DnCNN-PyTorch
    Zhang's GitHub Repo: https://github.com/cszn/DnCNN
    Reference: https://github.com/SaoYan/DnCNN-PyTorch/blob/6b0804951484eadb7f1ea24e8e5c9ede9bea485b/utils.py#L26

    Notes: Works with 3 dimensions as opposed to 2 dimensions
    """

    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def apply_data_aug_for_patches(patches, mode_range=(0, 8)):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Randomly augments each patch once from a choice of 8 augmentations using SaoYan's data augmentations
    """

    print('')
    print('Applying Data Augmentations on Patches...')

    augmented_patches = copy.deepcopy(patches)

    # print(type(patches))                    # <class 'numpy.ndarray'>
    # print(patches.dtype)                    # float32
    # print(patches.shape)                    # (42000, 10, 40, 40) for patch_size=40 and stride=10

    for patch_num in range(patches.shape[0]):
        # print(patch_num)
        augmented_patches[patch_num] = sao_yan_data_augmentation(patches[patch_num], mode=np.random.randint(mode_range[0], mode_range[1]))

    # show_image(augmented_patches[54][0])         # Load slices instead of patches to see augmentations clearly

    # print(type(patches))                    # <class 'numpy.ndarray'>
    # print(patches.dtype)                    # float32
    # print(patches.shape)                    # (42000, 10, 40, 40) for patch_size=40 and stride=10

    print('Finished augmenting', patches.shape[0], 'patches')

    return augmented_patches


def python_to_matlab_3d(data):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Convert from Python to Matlab image format for 3 dimensions
    """

    # Input = channel x height (num of rows) x width (num of cols)

    data = np.transpose(data, (0, 2, 1))    # data = channel x width (num of cols) x height (num of rows)
    data = np.transpose(data, (2, 1, 0))    # data = height (num of rows) x width (num of cols) x channel

    return data


def matlab_to_python_3d(data):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Convert from Matlab to Python image format for 3 dimensions
    """

    # Input = height (num of rows) x width (num of cols) x channel

    data = np.transpose(data, (2, 1, 0))    # data = channel x width (num of cols) x height (num of rows)
    data = np.transpose(data, (0, 2, 1))    # data = channel x height (num of rows) x width (num of cols)

    return data


def python_to_matlab_4d(data):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Convert from Python to Matlab image format for 4 dimensions
    """

    # Input = batch x channel x height (num of rows) x width (num of cols)

    data = np.transpose(data, (0, 1, 3, 2))     # data = batch x channel x width (num of cols) x height (num of rows)
    data = np.transpose(data, (3, 2, 1, 0))     # data = height (num of rows) x width (num of cols) x channel x batch

    return data


def matlab_to_python_4d(data):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Convert from Matlab to Python image format for 4 dimensions
    """

    # Input = height (num of rows) x width (num of cols) x channel x batch

    data = np.transpose(data, (3, 2, 1, 0))     # data = batch x channel x width (num of cols) x height (num of rows)
    data = np.transpose(data, (0, 1, 3, 2))     # data = batch x channel x height (num of rows) x width (num of cols)

    return data


def resize_images_with_batches(dataset, height, width):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Resize Python images with 4 dimensions using cv2
    """

    resized_dataset = []

    # -- Convert to Matlab format
    # print(dataset.shape)            # (8, 10, 230, 230)
    dataset = python_to_matlab_4d(dataset)  # Convert to height (num of rows) x width (num of cols) x channel x batch
    # print(dataset.shape)            # (230, 230, 10, 8)

    for batch_num in range(dataset.shape[3]):
        # -- Resize image
        resized_image = cv2.resize(src=dataset[..., batch_num], dsize=(height, width), interpolation=cv2.INTER_CUBIC)

        # -- Convert back to Python format
        # print(resized_image.shape)                               # (224, 224, 10)
        resized_image = matlab_to_python_3d(resized_image)
        # print(resized_image.shape)                               # (10, 224, 224)

        resized_dataset.append(resized_image)

    resized_dataset = np.asarray(resized_dataset)

    return resized_dataset


def export_to_onnx(net, ch, rows, cols, save_path):

    """
    Modified copy of code from PyTorch tutorial
    Instructions for exporting PyTorch model to ONNX and importing to Matlab: https://uk.mathworks.com/matlabcentral/answers/607941-how-to-load-a-fully-connected-pytorch-model-trained-model-into-matlab
    Code from PyTorch tutorial: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

    Code modified by: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Export PyTorch model to ONNX format
    """

    print('Exporting PyTorch Model to ONNX Format ...')

    batch_size = 1
    torch_model = net

    # -- Input to the model
    # x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)      # Original line
    x = torch.randn(batch_size, ch, rows, cols, requires_grad=True)     # KF - For anything - user defined parameters

    # -- Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      # "super_resolution.onnx",  # where to save the model (can be a file or file-like object)     # Original line
                      save_path,  # where to save the model (can be a file or file-like object)                     # KF line
                      export_params=True,  # store the trained parameter weights inside the model file
                      # opset_version=10,  # the ONNX version to export the model to                            # Original Line
                      opset_version=9,  # the ONNX version to export the model to                               # KF line
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print('Finished Exporting PyTorch Model to ONNX Format')

    return


def calculate_metrics_and_view(i, view_mode, view_clim, gt, noisy, outputs):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Compare ground-truth, noisy inputs and network outputs for each channel of a slice
    """

    # -- If inputs are type tensor then convert to numpy array metrics can be calculated
    if isinstance(outputs, torch.Tensor):
        gt_np = gt.detach().numpy()
        noisy_np = noisy.detach().numpy()
        outputs_np = outputs.detach().numpy()
    else:
        gt_np = gt
        noisy_np = noisy
        outputs_np = outputs

    # -- Iterate through channels of slice i
    for j in range(len(outputs[0])):
        print('')
        print('Slice:', i + 1, 'TSMI:', j + 1)

        if view_mode == 'view':

            # -- Calculate PSNR
            # Note: PSNR doesn't work without the data_range. SSIM has it for consistency
            print('PSNR (GT and Noisy):', psnr(gt_np[0][j], noisy_np[0][j], data_range=gt_np.max() - gt_np.min()))
            print('PSNR (GT and Inferred):', psnr(gt_np[0][j], outputs_np[0][j], data_range=gt_np.max() - gt_np.min()))

            # -- Calculate SSIM
            print('SSIM (GT and Noisy):', ssim(gt_np[0][j], noisy_np[0][j], data_range=gt_np.max() - gt_np.min()))
            print('SSIM (GT and Inferred):', ssim(gt_np[0][j], outputs_np[0][j], data_range=gt_np.max() - gt_np.min()))
            print('SSIM (GT and Noisy) (No Data Range):', ssim(gt_np[0][j], noisy_np[0][j]))
            print('SSIM (GT and Inferred) (No Data Range):', ssim(gt_np[0][j], outputs_np[0][j]))

            # -- Compare ground-truth, noisy inputs and network outputs of a channel
            compare_images(gt_np[0][j], noisy_np[0][j], outputs_np[0][j], view_clim)

    return
