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

# -- For scale_image_with_batch and rescale_image_with_batch
global global_scale_image_batch_x_min_values
global global_scale_image_batch_x_max_values
global_scale_image_batch_x_min_values = []
global_scale_image_batch_x_max_values = []
global global_undo_scale_image_batch_out_of_bounds_lower
global global_undo_scale_image_batch_out_of_bounds_upper
global_undo_scale_image_batch_out_of_bounds_lower = 0
global_undo_scale_image_batch_out_of_bounds_upper = 0


def scale_image_with_batch(input_data, lower_bounds, upper_bounds, debug_mode):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Normalise images stored as batches with the ability to check images are correctly normalised
    """

    print('')
    print(str('Normalizing images with batches from ' + str(lower_bounds) + ' to ' + str(upper_bounds) + ' ...'))

    # -- Clear min and max value variables so this function is ready to be used again
    global global_scale_image_batch_x_min_values
    global global_scale_image_batch_x_max_values
    global_scale_image_batch_x_min_values = []
    global_scale_image_batch_x_max_values = []

    # -- Need to copy input_data to prevent it from being modified in main scope. Modifying input_data
    # within this function affects input_data (the data input into this function) outside this function.
    scaled_data = copy.deepcopy(input_data)

    if debug_mode == 'on':
        print('')
        print('Scale_Batch Input Shape:', input_data.shape)      # Input shape = (15, 10, 230, 230)

    t_min = lower_bounds
    t_max = upper_bounds

    # -- Iterate through batches
    for batch in range(len(input_data)):

        if debug_mode == 'on':
            print('')
            print('Batch:', batch)

        # -- Find min and max values of batch
        x_input_min = np.amin(input_data[batch])
        x_input_max = np.amax(input_data[batch])

        # -- Perform Min-Max Scaling: x_scaled = (((x - x_min) / (x_max - x_min)) * (t_max - t_min)) + t_min
        if debug_mode == 'on':
            # print('Data before scaling:', input_data[batch])
            print('Data before scaling, Min:', np.amin(input_data[batch]))
            print('Data before scaling, Max:', np.amax(input_data[batch]))
        if (x_input_max - x_input_min) == 0:
            # - To prevent division by 0
            scaled_data[batch] = input_data[batch] * 0
        else:
            scaled_data[batch] = (((input_data[batch] - x_input_min) / (x_input_max - x_input_min)) * (t_max - t_min)) + t_min
        if debug_mode == 'on':
            # print('Data after scaling:', scaled_data[batch])
            print('Data after scaling, Min:', np.amin(scaled_data[batch]))
            print('Data after scaling, Max:', np.amax(scaled_data[batch]))

        # -- Save original min and max values for UnScaling later
        global_scale_image_batch_x_min_values.append(x_input_min)
        global_scale_image_batch_x_max_values.append(x_input_max)
        if debug_mode == 'on':
            print('Saved Min Values:', global_scale_image_batch_x_min_values)
            print('Saved Max Values:', global_scale_image_batch_x_max_values)

    print('Finished normalizing images with batches')

    return scaled_data


def rescale_image_with_batch(input_data, lower_bounds, upper_bounds, debug_mode, x_all_min, x_all_max):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022

    Description: Undo Normalisation of images stored as batches with the ability to check images are correctly
                 un-normalised. Out of bounds warnings shows that the CNN outputs are outside the 0 to 1 expected range
    """

    # -- Need to copy input_data to prevent it from being modified in main scope. Modifying input_data
    # within this function affects input_data (the data input into this function) outside this function.
    rescaled_data = copy.deepcopy(input_data)

    # -- Clear lower and upper bounds variables so this function is ready to be used again
    global global_undo_scale_image_batch_out_of_bounds_lower
    global global_undo_scale_image_batch_out_of_bounds_upper
    global_undo_scale_image_batch_out_of_bounds_lower = 0
    global_undo_scale_image_batch_out_of_bounds_upper = 0

    if debug_mode == 'on':
        print('')
        print('Rescale_Batch Input Shape:', input_data.shape)  # Input shape = (15, 10, 230, 230)
        print('x_all_min:', x_all_min)
        print('x_all_max:', x_all_max)

    t_min = lower_bounds
    t_max = upper_bounds

    # -- Iterate through batches
    for batch in range(len(input_data)):

        if debug_mode == 'on':
            print('')
            print('Batch:', batch)

        # -- Get saved min and max values of batch
        x_min = x_all_min[batch]
        x_max = x_all_max[batch]
        if debug_mode == 'on':
            print('Saved Min Value:', x_min)
            print('Saved Max Value:', x_max)

        if debug_mode == 'on':
            # print('Data before scaling is undone:', input_data[batch])
            print('Data before scaling is undone, Min:', np.amin(input_data[batch]))
            print('Data before scaling is undone, Max:', np.amax(input_data[batch]))

        # -- Display warning and count how many times values predicted from network are < 0 or > 1
        # I found that the values predicted by the network could be < 0 or > 1 which will give incorrect values
        # when Min-Max Scaling is undone.
        if np.amin(input_data[batch]) < t_min:
            if debug_mode == 'on':
                print('***WARNING: Min Value is < Lower Bounds:', np.amin(input_data[batch]))
            global_undo_scale_image_batch_out_of_bounds_lower += 1
        if np.amax(input_data[batch]) > t_max:
            if debug_mode == 'on':
                print('***WARNING: Max Value is > Upper Bounds:', np.amax(input_data[batch]))
            global_undo_scale_image_batch_out_of_bounds_upper += 1

        # -- Perform Undo Min-Max Scaling: x = (((x_scaled - t_min) / (t_max - t_min)) * (x_max - x_min)) + x_min
        rescaled_data[batch] = (((input_data[batch] - t_min) / (t_max - t_min)) * (x_max - x_min)) + x_min

        if debug_mode == 'on':
            # print('Data after scaling is undone:', rescaled_data[batch])
            print('Data after scaling is undone, Min:', np.amin(rescaled_data[batch]))
            print('Data after scaling is undone, Max:', np.amax(rescaled_data[batch]))

    print('')
    print('Number of values less than lower bounds in rescale_image_with_batch():',
          global_undo_scale_image_batch_out_of_bounds_lower)
    print('Number of values greater than upper bounds in rescale_image_with_batch():',
          global_undo_scale_image_batch_out_of_bounds_upper)

    return rescaled_data
