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
import numpy as np
import pickle
import re                           # For numerical_sort function from website


def numerical_sort(value):

    """
    Author: Martijn Pieters
    Code Reference: https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
    Description: Function to be used for sorted(..., key=HERE), to sort and put filenames in correct order
    """

    numbers = re.compile(r'(\d+)')              # This was outside the function in the code from the website
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])

    return parts


def load_real_fisp_float64(path, sort=False):

    """
    Author: Ketan Fatania
    Affiliation: University of Bath
    Email: kf432@bath.ac.uk
    GitHub: https://github.com/ketanfatania/QMRI-PnP-Recon-POC
    Date: --/09/2020 to --/02/2022
    """

    files = os.listdir(path)
    # print(files)                                      # Reads files in a random order
    if sort:
        files = sorted(files, key=numerical_sort)
        # print(files)                                  # Files are now correctly sorted

    x_seq_np_all = []
    for file in files:
        with open(path + file, 'rb') as full_path:
            print(path + file)
            if file == '.DS_Store':
                pass                                    # Ignore .DS_Store file on Mac
            else:
                x_seq_np = pickle.load(full_path)
                x_seq_np_all.append(x_seq_np)

    # print(type(x_seq_np_all))                         # <class 'list'>
    # print(np.asarray(x_seq_np_all).shape)             # (7, 15, 10, 230, 230) for training data and (1, 15, 10, 230, 230) for test data

    # - Remove volunteers dimension
    x_slices = []
    for i in range(len(x_seq_np_all)):
        x_slices.extend(x_seq_np_all[i])

    # print(type(x_slices))                             # <class 'list'>
    # print(np.asarray(x_slices).shape)                 # (105, 10, 230, 230) for training data and (15, 10, 230, 230) for test data

    x_ground_truth = np.asarray(x_slices)               # Convert to numpy.ndarray

    # print(type(x_ground_truth))                       # <class 'numpy.ndarray'>
    # print(np.asarray(x_ground_truth).shape)           # (105, 10, 230, 230) for training data and (15, 10, 230, 230) for testing data

    return x_ground_truth
