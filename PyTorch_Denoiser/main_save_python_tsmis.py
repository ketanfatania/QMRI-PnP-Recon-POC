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

import os
import re
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


def show_image(image):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022
    """

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar(orientation='horizontal')
    plt.show()


def numerical_sort(value):

    """
    Author: Martijn Pieters
    Code Reference: https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
    Description: Function to be used for sorted(..., key=HERE), to sort and put filenames in correct order
    """

    numbers = re.compile(r'(\d+)')          # This was outside the function in the code from the website
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def examine_data(slice_path):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022
    """

    print(slice_path)

    data = scio.loadmat(slice_path)
    print(type(data))                   # <class 'dict'>
    print(data.keys())                  # dict_keys(['__header__', '__version__', '__globals__', 'X'])
    print(data['__header__'])
    print(data['__version__'])
    print(data['__globals__'])
    print(data['X'])

    data_slice = data['X']
    print(type(data_slice))             # <class 'numpy.ndarray'>
    print(data_slice.dtype)             # float64
    print(data_slice.shape)             # (230, 230, 10)
    show_image(data_slice[:, :, 0])

    data_slice_transposed = np.transpose(data_slice, (2, 1, 0))                         # Image is rotated
    data_slice_transposed = np.transpose(data_slice_transposed, (0, 2, 1))              # Image is now in correct orientation
    print(data_slice_transposed.shape)                                                  # (10, 230, 230)
    show_image(data_slice_transposed[0])

    data_slice_transposed_back = np.transpose(data_slice_transposed, (0, 2, 1))         # Image is rotated
    data_slice_transposed_back = np.transpose(data_slice_transposed_back, (2, 1, 0))    # Image is now in correct orientation
    print(data_slice_transposed_back.shape)                                             # (230, 230, 10)
    show_image(data_slice_transposed_back[:,:,0])

    print((data_slice.reshape(-1) == data_slice_transposed_back.reshape(-1)).all())     # True when transposed back


def ready_real_data(args, select_channels=False, channels_to_save=None):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022
    """

    # -- Initialise variables
    scan_type = args.scan_type
    cut = args.cut
    num_slices = args.num_slices
    training_data_input_path = args.training_data_input_path
    testing_data_input_path = args.testing_data_input_path
    training_data_output_path = args.training_data_output_path
    testing_data_output_path = args.testing_data_output_path
    matlab_train_test_split = args.matlab_train_test_split
    python_train_test_split = args.python_train_test_split

    # -- Gather all training and testing filenames
    files_train = os.listdir(training_data_input_path)
    files_test = os.listdir(testing_data_input_path)
    files_train.extend(files_test)
    files = files_train
    print(files)                                # Files are not in correct order

    files = sorted(files, key=numerical_sort)
    print(files)                                # Files are now correctly sorted

    # -- Initialise variables for loop
    slices_np_all = []
    files_check = []
    slice_counter = 0
    vol_counter = 0

    # -- Loop through all slices
    for file in files:
        files_check.append(file)
        print(file)

        # - Ignore .DS_Store file
        if file == '.DS_Store':
            pass

        else:
            # - Choose correct path for current Matlab data filename
            if (vol_counter+1) < matlab_train_test_split:
                path = training_data_input_path
            else:
                path = testing_data_input_path

            # - Load Matlab file
            data = scio.loadmat(path + file)
            data_slice = data['X']
            print(data_slice.dtype)                 # float64
            print(data_slice.shape)                 # (230, 230, 10)

            # - Transpose Matlab data to Python format
            data_slice_transposed = np.transpose(data_slice, (2, 1, 0))
            data_slice_transposed = np.transpose(data_slice_transposed, (0, 2, 1))
            print(data_slice_transposed.dtype)                  # float64
            print(data_slice_transposed.shape)                  # (10, 230, 230)

            if select_channels == True:
                if channels_to_save == 1:
                    num_rows = data_slice_transposed.shape[1]
                    num_cols = data_slice_transposed.shape[2]
                    data_slice_final = data_slice_transposed[0].reshape(channels_to_save, num_rows, num_cols)
                    print(data_slice_final.dtype)               # float64
                    print(data_slice_final.shape)               # (1, 230, 230)
                else:
                    data_slice_final = data_slice_transposed[0:channels_to_save]
                    print(data_slice_final.dtype)               # float64
                    print(data_slice_final.shape)               # (4, 230, 230)
                slices_np_all.append(data_slice_final)
                print('slices_np_all:', np.asarray(slices_np_all).shape)
            else:
                # - Save all channels
                slices_np_all.append(data_slice_transposed)

            slice_counter = slice_counter + 1
            # - Save volunteer/subject data after iterating through all Matlab slices for volunteer/subject
            if slice_counter == num_slices:
                slice_counter = 0
                vol_counter = vol_counter + 1

                vol_data = np.asarray(slices_np_all)
                print(files_check)                      # Print files included in vol_data
                print(type(vol_data))
                print(vol_data.dtype)
                print(vol_data.shape)                   # (15, 10, 230, 230)

                # - Get number of channels
                vol_num_ch = vol_data.shape[1]
                print(vol_num_ch)                       # 10

                # - Save Python training and testing data
                if vol_counter < python_train_test_split:
                    with open(str(training_data_output_path + 'vol' + str(vol_counter) + '_real_' + scan_type + '_cut' + str(cut) + '_numpyfloat64_' + str(vol_num_ch) + 'channels.pkl'), 'wb') as f:
                        pickle.dump(vol_data, f)
                else:
                    with open(str(testing_data_output_path + 'vol' + str(vol_counter) + '_real_' + scan_type + '_cut' + str(cut) + '_numpyfloat64_' + str(vol_num_ch) + 'channels.pkl'), 'wb') as f:
                        pickle.dump(vol_data, f)

                # - Reset variables
                files_check = []
                slices_np_all = []


def check_ready_real_data(path, slice_num, channel_num):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022
    """

    with open(path, 'rb') as f:
        vol_data = pickle.load(f)

    print(type(vol_data))               # <class 'numpy.ndarray'>
    print(vol_data.dtype)               # float64
    print(vol_data.shape)               # (15, 10, 230, 230)
    show_image(vol_data[slice_num][channel_num])


if __name__ == '__main__':

    args = argparse.ArgumentParser().parse_args()

    # ------------------------------------------
    # -- Acquisition Settings
    args.scan_type = 'fisp'
    args.cut = 3

    # -- Number of Slices per Subject in Matlab Data
    args.num_slices = 15

    # -- Train and Test Split
    args.matlab_train_test_split = 8        # The volunteer/subject number at which Matlab testing data begins
    args.python_train_test_split = 8        # The volunteer/subject number at which creation of Python testing data begins
    # ------------------------------------------

    # ------------------------------------------
    # -- Select Data to Examine
    slice_path = str('../datasets/synth_tsmis/real_' + args.scan_type + '_cut' + str(args.cut) + '_tsmis/training_data/vol1s1.mat')
    slice_path = str('../datasets/synth_tsmis/real_' + args.scan_type + '_cut' + str(args.cut) + '_tsmis/testing_data/vol8s1.mat')

    # -- Select Matlab Data to Convert to Python Data
    args.training_data_input_path = str('../datasets/synth_tsmis/real_' + args.scan_type + '_cut' + str(args.cut) + '_tsmis/training_data/')
    args.testing_data_input_path = str('../datasets/synth_tsmis/real_' + args.scan_type + '_cut' + str(args.cut) + '_tsmis/testing_data/')

    # -- Select where to put Python Data
    args.training_data_output_path = str('./datasets/real_' + args.scan_type + '_cut' + str(args.cut) + '_float64_pkl/training_data/')
    args.testing_data_output_path = str('./datasets/real_' + args.scan_type + '_cut' + str(args.cut) + '_float64_pkl/testing_data/')

    # -- Create output directories if they do not exist
    # - Check if training data output exists
    if not os.path.exists(args.training_data_output_path):
        os.makedirs(args.training_data_output_path)
    # - Check if testing data output exists
    if not os.path.exists(args.testing_data_output_path):
        os.makedirs(args.testing_data_output_path)

    # -- Select Readied Real Data to Check
    # vol_path = str('./datasets/real_' + args.scan_type + '_cut' + str(args.cut) + '_float64_pkl/training_data/vol1_real_fisp_cut3_numpyfloat64_10channels.pkl')
    vol_path = str('./datasets/real_' + args.scan_type + '_cut' + str(args.cut) + '_float64_pkl/testing_data/vol8_real_fisp_cut3_numpyfloat64_10channels.pkl')
    # ------------------------------------------

    # # ------------------------------------------
    # # -- Examine Matlab TSMIs
    # examine_data(slice_path=slice_path)               # Examine TSMI data generated from QMaps using Matlab
    # exit()
    # # ------------------------------------------

    # ------------------------------------------
    # -- Prepare abs(complex) Python TSMIs with all channels from Matlab TSMIs
    # Note: If select_channels = False then all channels will be saved regardless of what channels_to_save is set to
    ready_real_data(args, select_channels=False)

    # # -- Prepare abs(complex) Python TSMIs with specified number of channels from Matlab TSMIs
    # ready_real_data(args, select_channels=True, channels_to_save=1)

    # # -- Check if data has been saved correctly
    # check_ready_real_data(path=vol_path, slice_num=0, channel_num=0)
    # ------------------------------------------
