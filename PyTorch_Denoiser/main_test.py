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
import argparse
import datetime

import torch

import data
from utils import export_to_onnx
from utils import calculate_metrics_and_view
from scale_rescale import rescale_image_with_batch


# -- Import code from Zhang et al. DPIR (DRUNet): https://github.com/cszn/DPIR
# @article{zhang2021plug,
#   title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
#   author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
#   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   year={2021}
# }
from zhang_dpir_testing_code.network_unet import UNetRes        # Original code with only import basicblock.py location changed


if __name__ == '__main__':

    def test_init_settings():

        args = argparse.ArgumentParser().parse_args()
        args_data_test_settings = argparse.ArgumentParser().parse_args()
        args_main_test_settings = argparse.ArgumentParser().parse_args()

        # ----------------- Hardware Settings ----------------
        # -- Set Device to use
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print('Device: ', device)
        args_main_test_settings.device = device

        # -- Set dataloader settings
        args.pin_memory = False
        args.num_workers_test = 0
        # ----------------- Hardware Settings ----------------

        # ----------------- Float32 or Float64 Settings ----------------
        # This controls the CNN network type
        # This also controls the dtype of data used
        # This also controls the dtype of noise used
        # Note: Float64 data requires that args.float32or64_mode = '64' otherwise will throw an error.
        #       Fixed by enforcing float32 or float64 in data.py

        args.float32or64_mode = '32'            # '32' or '64'

        if args.float32or64_mode == '32':
            print('Using default torch.set_default_dtype(torch.float32)')
        elif args.float32or64_mode == '64':
            print('Setting torch.set_default_dtype(torch.float64)')
            torch.set_default_dtype(torch.float64)
        # ----------------- Float32 or Float64 Settings ----------------

        # ----------------- General Settings ----------------
        # -- Acquisition Settings
        args.scan_type = 'fisp'                                 # 'fisp'
        args.cut = '3'                                          # 0, 1, 2, 3 or 4

        # -- Patches Mode
        args.patches_mode = 'off'                               # 'on' or 'off' - 'on' not used for ISBI paper

        # -- Select Batch Size
        args.test_batch_size = 1                            # Batch size for patches off
        args.test_patch_batch_size = 81                     # Batch size for patches on - not used in ISBI 2022 paper

        # -- To Export Test PyTorch Model to ONNX
        args.export_onnx_model = 'off'        # 'on' or 'off'
        # ----------------- General Settings ----------------

        # ----------------- Denoiser Settings ----------------
        # -- Select Network Architecture
        args.network_architecture = 'DRUNet_Single_Level'        # 'DRUNet_Single_Level' or 'DRUNet_Multi_Level'

        # -- Set channels, rows and cols for architecture type
        if args.network_architecture == 'DRUNet_Single_Level':
            args.num_channels = 10          # For export_to_onnx function
            args.num_rows = 224             # For export_to_onnx function and data.py
            args.num_cols = 224             # For export_to_onnx function and data.py
        elif args.network_architecture == 'DRUNet_Multi_Level':
            args.num_channels = 11          # For export_to_onnx function
            args.num_rows = 224             # For export_to_onnx function and data.py
            args.num_cols = 224             # For export_to_onnx function and data.py

        # -- Set variables for use in data.py
        args_data_test_settings.num_rows = args.num_rows
        args_data_test_settings.num_cols = args.num_cols
        # ----------------- Denoiser Settings ----------------

        # ----------------- Path Settings ----------------
        # -- Directory of Test Models to Load
        args.load_test_model_dir = str('./final_pt_models/real_' + args.scan_type + '_cut' + str(args.cut) + '_pt_models/')

        # -- Filenames of Test Models to Load
        args.load_test_model_single_level_filename = 'pytorch_model_drunet_10ch_500epochs.pt'
        args.load_test_model_multi_level_filename = 'pytorch_model_drunet_11ch_500epochs.pt'

        # -- Directory of where to save ONNX Models
        args.save_onnx_model_dir = str('../onnx_models/real_' + args.scan_type + '_cut' + str(args.cut) + '_onnx_models/')

        # -- Filenames of ONNX Files to save
        # - Add date and time to start of ONNX filename to prevent overwriting of existing files
        current_datetime = datetime.datetime.now()
        save_path_datetime = str(current_datetime.date()) + '_' + str(current_datetime.strftime('%H-%M-%S'))
        # - Create ONNX filename
        args.save_onnx_model_single_level_filename = str(str(save_path_datetime) + '_pytorch_model_drunet_10ch_500epochs.onnx')
        args.save_onnx_model_multi_level_filename = str(str(save_path_datetime) + '_pytorch_model_drunet_11ch_500epochs.onnx')

        # -- Create ONNX save directory if it does not exist
        # - Check if ONNX save directory exists
        if not os.path.exists(args.save_onnx_model_dir):
            os.makedirs(args.save_onnx_model_dir)

        # -- Set load test model path and save ONNX file path
        if args.network_architecture == 'DRUNet_Single_Level':
            args.test_model_path = str(args.load_test_model_dir + args.load_test_model_single_level_filename)
            args.onnx_save_path = str(args.save_onnx_model_dir + args.save_onnx_model_single_level_filename)
        elif args.network_architecture == 'DRUNet_Multi_Level':
            args.test_model_path = str(args.load_test_model_dir + args.load_test_model_multi_level_filename)
            args.onnx_save_path = str(args.save_onnx_model_dir + args.save_onnx_model_multi_level_filename)
        # ----------------- Path Settings ----------------

        # ----------------- Datasets Settings ----------------
        # -- Datasets Available
        # 'real_fisp_cut0_float64',
        # 'real_fisp_cut1_float64', 'real_fisp_cut2_float64',
        # 'real_fisp_cut3_float64', 'real_fisp_cut4_float64'
        # Note: Correct dtype for dataset is enforced in data.py
        args_data_test_settings.dataset = 'real_' + args.scan_type + '_cut' + args.cut + '_float64'

        # -- Select slice to test
        args_data_test_settings.slice_num = 9               # Note: 0 = slice 1

        # -- Set Testing Path
        args_data_test_settings.testing_data_path = str('./datasets/' + args_data_test_settings.dataset + '_pkl/testing_data/')

        # - Get settings to use Float32 or Float64 datatype
        args_data_test_settings.dataset_float32or64_mode = args.float32or64_mode  # '32' or '64'
        # ----------------- Datasets Settings ----------------

        # ----------------- Dataset Creation Settings ----------------
        # -- Select Data Creation Method in data.py
        # - Patches Off:
        #       DRUNet_Single_Level = crop to 224x224, normalize 0to1, generate noisy data
        #       DRUNet_Multi_Level = crop to 224x224, normalize 0to1, generate noisy data with noise map
        # - Patches On:
        #       not_used_for_ISBI_2022_paper
        if args.patches_mode == 'off':
            # args_data_test_settings.patches_off_mode = 'DRUNet_Single_Level'              # Manually select test data creation method
            args_data_test_settings.patches_off_mode = args.network_architecture            # Select using network architecture selection
        elif args.patches_mode == 'on':
            args_data_test_settings.patches_on_mode = 'not_used_for_ISBI_2022_paper'        # Manually select test data creation method
        # ----------------- Dataset Creation Settings ----------------

        # ----------------- Noise Settings ----------------
        # -- Set Gaussian Noise for Single-Level Denoiser and Multi-Level Denoiser
        args.gauss_mean = 0
        args.gauss_var = 0.0001                       # 0.0001
        args.gauss_std = args.gauss_var ** 0.5        # 0.0001var ** 0.5 = 0.01std
        args.gauss_std = 0.01                         # Note: overrides above. Use this for Noise Map too

        # -- Get settings for args.gauss_dtype to set noise dtype
        if args.float32or64_mode == '32':
            args.gauss_dtype = '32'
        elif args.float32or64_mode == '64':
            args.gauss_dtype = '64'

        # -- Get Noise Settings for Single-Level Denoiser and Multi-Level Denoiser in data.py
        args_data_test_settings.noise_gauss_mean = args.gauss_mean
        args_data_test_settings.noise_gauss_var = args.gauss_var
        args_data_test_settings.noise_gauss_std = args.gauss_std
        args_data_test_settings.noise_gauss_dtype = args.gauss_dtype
        # ----------------- Noise Settings ----------------

        # --------------- Undo-Normalization Settings --------------
        # -- Select Undo-Normalization Settings
        args.transforms_off_undo_scale_image = 'on'                 # 'on' or 'off'
        args.rescale_image_lower_bounds = 0
        args.rescale_image_upper_bounds = 1

        # -- Choose if outputs, x_gt, x_noisy_inputs are normalized back for quantitative and qualitative analysis
        # ground-truth is not normalized during data creation, so no need to normalize it back
        args_main_test_settings.patches_off_unscale_image_choice = [True, False, True]      # [True, False, True]
        # --------------- Undo-Normalization Settings --------------

        # -------------- Settings View Images and Show Metrics ----------------
        # -- View images and Show Metrics
        args.view_mode = 'view'    # 'view'

        # -- Set colour limits of compare_images() when viewing test images in 'view' mode
        args.view_clim = None                   # None = don't use plt.clim()
        # args.view_clim = 'gt'                 # 'gt' = use max and min colour limit of ground truth
        # args.view_clim = [-0.5, 0.5]          # set custom colour limits, e.g [-0.5, 0.5]
        # -------------- Settings View Images and Show Metrics ----------------

        test_denoiser(args, args_data_test_settings, args_main_test_settings)


    def test_denoiser(args, args_data_test_settings, args_main_test_settings):

        # -- Set Device to use
        device = args_main_test_settings.device

        # -- Initialise Testing Data
        print('')
        print(' -- Initialising Testing Data - Running data.TSMIDataset ...')
        test_dataset = data.TSMIDataset(mode='test', patches_mode=args.patches_mode,
                                        test_settings=args_data_test_settings)

        # -- Set batch size
        test_batch_size = args.test_patch_batch_size if args.patches_mode == 'on' else args.test_batch_size
        print('')
        print('Test Batch Size: ', test_batch_size)

        # -- Initialise DataLoader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers_test, pin_memory=args.pin_memory)

        # -- Choose Architecture
        # Initialise Net (For Zhang DRUNet - 10+1 Channel with Noise Map)
        # Note: "net" is sent to the "device"
        if args.network_architecture == 'DRUNet_Single_Level':
            # - Without Noise Map: Do not add 1 channel for noise map
            # Reference: Line below is from Zhang et al. DPIR code main_dpir_denoising.py and modified for 10 input channels
            net = UNetRes(in_nc=10, out_nc=10, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
        elif args.network_architecture == 'DRUNet_Multi_Level':
            # - With Noise Map: Add 1 channel to input channels to accommodate for FFDNet noise map
            # Reference: Line below is from Zhang et al. DPIR code main_dpir_denoising.py
            net = UNetRes(in_nc=10+1, out_nc=10, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
        else:
            print('Select a valid args.network_architecture choice')
            exit()
        print('')
        print(net)

        # -- Load and initialise saved model
        checkpoint = torch.load(f=args.test_model_path, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.eval()                                          # Enable evaluation mode
        print('Model Loaded:', args.test_model_path)
        print('Model Epoch:', epoch)
        print('Model Training Loss:', loss)

        # -- Use export_to_onnx in utils.py to export PyTorch model to ONNX format, so it can be loaded in Matlab later
        # Note: Need to send model in evaluation mode - net.eval() must be used before sending net through
        if args.export_onnx_model == 'on':
            export_to_onnx(net, ch=args.num_channels, rows=args.num_rows, cols=args.num_cols, save_path=args.onnx_save_path)
            exit()

        # -- Cycle through all testing data
        print('Start testing...')
        for i, test_data in enumerate(test_loader, 0):      # Cycle through all slices i = 0-14 for volunteer 8

            x_noisy_inputs, x_gt = test_data[0].to(device), test_data[1].to(device)

            outputs = net(x_noisy_inputs)

            # -- Code for patches_mode == 'on'
            if args.patches_mode == 'on':
                print('Not used in ISBI 2022 paper')
                exit()

            # -- Code for patches_mode == 'off'
            elif args.patches_mode == 'off':

                # -- Initialise variables
                regen_outputs = outputs
                regen_gt = x_gt
                regen_noisy = x_noisy_inputs

                # -- If inputs are type tensor convert to numpy array
                if isinstance(outputs, torch.Tensor):
                    regen_outputs = regen_outputs.detach().numpy()
                    regen_gt = regen_gt.detach().numpy()
                    regen_noisy = regen_noisy.detach().numpy()

                # -- Undo-Normalization
                if args.transforms_off_undo_scale_image == 'on':

                    print('UnScaling images with rescale_image_with_batch()...')

                    # - Undo-Normalization for outputs
                    if args_main_test_settings.patches_off_unscale_image_choice[0] == True:
                        regen_outputs = rescale_image_with_batch(input_data=regen_outputs, debug_mode='on',
                                                                 lower_bounds=args.rescale_image_lower_bounds,
                                                                 upper_bounds=args.rescale_image_upper_bounds,
                                                                 x_all_min=data.global_scale_image_batch_x_min_values_noisy,
                                                                 x_all_max=data.global_scale_image_batch_x_max_values_noisy)

                    # - Undo-Normalization for x_gt
                    if args_main_test_settings.patches_off_unscale_image_choice[1] == True:
                        regen_gt = rescale_image_with_batch(input_data=regen_gt, debug_mode='on',
                                                            lower_bounds=args.rescale_image_lower_bounds,
                                                            upper_bounds=args.rescale_image_upper_bounds,
                                                            x_all_min=data.global_scale_image_batch_x_min_values_gt,
                                                            x_all_max=data.global_scale_image_batch_x_max_values_gt)

                    # - Undo-Normalization for x_noisy_inputs
                    if args_main_test_settings.patches_off_unscale_image_choice[2] == True:
                        regen_noisy = rescale_image_with_batch(input_data=regen_noisy, debug_mode='on',
                                                               lower_bounds=args.rescale_image_lower_bounds,
                                                               upper_bounds=args.rescale_image_upper_bounds,
                                                               x_all_min=data.global_scale_image_batch_x_min_values_noisy,
                                                               x_all_max=data.global_scale_image_batch_x_max_values_noisy)

                # -- Calculate metrics and view channels of a slice
                calculate_metrics_and_view(i, args.view_mode, args.view_clim,
                                           regen_gt, regen_noisy, regen_outputs)

    test_init_settings()
