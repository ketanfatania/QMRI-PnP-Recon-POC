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
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import data
from utils import generate_noisy_data_fly
from utils import generate_noisy_data_fly_blind
from utils import generate_noise_map_fly

# -- Import testing code from Zhang et al. DPIR (DRUNet): https://github.com/cszn/DPIR
# @article{zhang2021plug,
#   title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
#   author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
#   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   year={2021}
# }
from zhang_dpir_testing_code.network_unet import UNetRes        # Original code with only import basicblock.py location changed


if __name__ == '__main__':

    def train_init_settings():

        args = argparse.ArgumentParser().parse_args()
        args_data_train_settings = argparse.ArgumentParser().parse_args()
        args_main_train_settings = argparse.ArgumentParser().parse_args()

        # ----------------- Hardware Settings ----------------
        # -- Set Device to use
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print('Device: ', device)
        args_main_train_settings.device = device

        # -- Loading all Data to Device
        # Note: Noise is generated on the fly on cpu regardless of this setting and then sent to device to be added on
        # the device
        #       off - data will be loaded to the device in batches during the training loop
        #       on - all data is loaded to the device in data.py
        args_data_train_settings.load_all_data_to_gpu = 'off'   # 'on' or 'off'

        # -- Set dataloader settings
        args.pin_memory = False
        args.num_workers_train = 4                              # Set to 0 if it gives issues
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

        # -- Select Network Architecture
        args.network_architecture = 'DRUNet_Single_Level'        # 'DRUNet_Single_Level' or 'DRUNet_Multi_Level'

        # -- Patches Mode
        args.patches_mode = 'on'                                # 'on' or 'off' - 'off' not used for ISBI paper training
        args_data_train_settings.patch_size = 128
        args_data_train_settings.patch_stride = 17
        # - Resize Image Scales Before Patching
        args_data_train_settings.resize_scales = [1, 0.9, 0.8, 0.7]

        # -- Select Batch Size
        args.batch_size = 16                # Batch size for patches off
        args.patch_batch_size = 16          # Batch size for patches on

        # -- Select Learning Rate
        args.lr = 1e-4

        # -- Weights and Bias Settings
        # pytorch_default = Use PyTorch's default weights and bias initialisation method
        args.weights_bias_init_method = 'pytorch_default'               # 'pytorch_default'

        # -- Scheduler Settings
        # - Zhang et al. writes that they us a scheduler which halves the LR every 100,000 iterations
        #       - My calculation for Patch Size 128, stride 17, scaling [1, 0.9, 0.8, 0.7], batch size 16 which
        #         gives training dataset size of 9870:
        #             9870images(patches) / 16batchsize ~= 616steps_per_epoch. So 100,000steps / 616steps = 162epochs
        # Note: Need to half LR 8 times only (not specific to above settings)
        args.scheduler_milestones = [162, 324, 486, 648, 810, 972, 1134, 1296]
        args.scheduler_gamma = 0.5

        # -- Select Max Number of Epochs
        args.epochs = 500 #1458 #DRUNet-patchsize128stride17,batchsize16,scaling[1,0.9,0.8,0.7]

        # -- Create a Checkpoint Every This Many Number of Epochs
        args.checkpoints_epochs = 50

        # -- Output Paths
        args.checkpoints_dir = './checkpoints/'
        args.final_model_dir = './final_pt_models/real_' + args.scan_type + '_cut' + str(args.cut) + '_pt_models/'

        # -- Create output directories if they do not exist
        # - Check if checkpoints directory exists
        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)
        # - Check if final model directory exists
        if not os.path.exists(args.final_model_dir):
            os.makedirs(args.final_model_dir)

        # -- Resume Training Settings
        args.resume_training = 'off'  # on or off
        args.resume_training_path = './checkpoints/put_checkpoint_file_name_here.pt'
        args.resume_training_sumwri_dir = './runs/put_session_folder_name_here/'
        # ----------------- General Settings ----------------

        # ----------------- Datasets Settings ----------------
        # -- Datasets Available
        # 'real_fisp_cut0_float64',
        # 'real_fisp_cut1_float64', 'real_fisp_cut2_float64',
        # 'real_fisp_cut3_float64', 'real_fisp_cut4_float64'
        # Note: Correct dtype for dataset is enforced in data.py
        args_data_train_settings.dataset = 'real_' + args.scan_type + '_cut' + args.cut + '_float64'

        # -- Set Training Path
        args_data_train_settings.training_data_path = str('./datasets/' + args_data_train_settings.dataset + '_pkl/training_data/')

        # -- Get settings to use Float32 or Float64 datatype
        args_data_train_settings.dataset_float32or64_mode = args.float32or64_mode        # '32' or '64'
        # ----------------- Datasets Settings ----------------

        # ----------------- Noise Settings ----------------
        # -- Gaussian Noise for Single-Level Denoiser
        args.gauss_mean = 0
        args.gauss_var = 0.0001                     # 0.0001
        args.gauss_std = args.gauss_var ** 0.5      # var=0.0001 is equal to std=0.01
        args.gauss_std = 0.01                       # OVERRIDES ABOVE LINE

        # -- Gaussian Noise for Multi-Level Denoiser
        args.gauss_blind_mean = 0
        # args.gauss_blind_var = []                 # Not used, only for completeness - use std instead
        args.gauss_blind_std = [0.0001, 1]          # Uniformly sample values from [0.0001, 1]

        # -- Get settings for args.gauss_dtype to set noise dtype
        if args.float32or64_mode == '32':
            args.gauss_dtype = '32'
        elif args.float32or64_mode == '64':
            args.gauss_dtype = '64'

        # -- Get Noise Settings for Single-Level Denoiser in main.py
        args_main_train_settings.noise_gauss_mean = args.gauss_mean
        args_main_train_settings.noise_gauss_var = args.gauss_var
        args_main_train_settings.noise_gauss_std = args.gauss_std
        args_main_train_settings.noise_gauss_dtype = args.gauss_dtype

        # -- Get Noise Settings for Multi-Level Denoiser in main.py
        args_main_train_settings.noise_gauss_blind_mean = args.gauss_blind_mean
        # args_main_train_settings.noise_gauss_blind_var = args.gauss_blind_var     # Not used, here for completeness - use std instead
        args_main_train_settings.noise_gauss_blind_std = args.gauss_blind_std
        args_main_train_settings.noise_gauss_blind_dtype = args.gauss_dtype

        # -- Noise Names for Checkpoints, Summary Writer and Final Models
        if args.network_architecture == 'DRUNet_Single_Level':
            args.checkpoint_std = str(args.gauss_std)
            args.checkpoint_mean = str(args.gauss_mean)
        elif args.network_architecture == 'DRUNet_Multi_Level':
            args.checkpoint_std = str(str(args.gauss_blind_std[0]) + 'to' + str(args.gauss_blind_std[1]))
            args.checkpoint_mean = str(args.gauss_blind_mean)
        else:
            print('Defaulting to Using Single-Level Noise STD and Mean for Naming Checkpoints')
            args.checkpoint_std = str(args.gauss_std)
            args.checkpoint_mean = str(args.gauss_mean)
        # ----------------- Noise Settings ----------------

        # ----------------- Dataset Creation and On-The-Fly Settings ----------------
        # -- Select data creation mode and on the fly mode
        #       ISBI_2022_Paper - Data creation and noise on the fly as described in the ISBI 2022 paper
        if args.patches_mode == 'on':
            args_data_train_settings.patches_on_mode = 'ISBI_2022_Paper'
            args.gen_noisy_data_fly_mode = args_data_train_settings.patches_on_mode
        elif args.patches_mode == 'off':
            args_data_train_settings.patches_off_mode = 'not_used_for_ISBI_2022_paper'
            args.gen_noisy_data_fly_mode = args_data_train_settings.patches_off_mode
        # ----------------- Dataset Creation and On-The-Fly Settings ----------------

        train_denoiser(args, args_data_train_settings, args_main_train_settings)


    def train_denoiser(args, args_data_train_settings, args_main_train_settings):

        # -- Set Device to use
        device = args_main_train_settings.device

        # -- Initialise Training Data
        print('')
        print(' -- Initialising Training Data - Running data.TSMIDataset ...')
        train_dataset = data.TSMIDataset(mode='train', patches_mode=args.patches_mode,
                                         train_settings=args_data_train_settings,
                                         device=device)

        # -- Set batch size
        batch_size = args.patch_batch_size if args.patches_mode == 'on' else args.batch_size
        print('')
        print('Batch Size: ', batch_size)

        # -- Initialise DataLoader
        # Note: torch.utils.data.DataLoader() automatically converts data to type torch.Tensor from numpy.ndarray
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, num_workers=args.num_workers_train, pin_memory=args.pin_memory)

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

        # -- Initialise network for training
        net.train()
        print('')
        print(net)

        # -- Initialise Weights and Bias for layers in network
        # Note: No bias was used during the creation of the "net", so bias is not initialised.
        print('')
        if args.weights_bias_init_method == 'pytorch_default':
            # PyTorch Defaults: Weights = init.Kaiming_Uniform_, Bias = init.Uniform_
            print('Using PyTorch default weights and bias initialisation...')

        # -- Initialise Loss Function
        criterion = nn.L1Loss()
        print('')
        print(criterion)

        # -- Initialise Optimizer
        optimizer = optim.Adam(params=net.parameters(), lr=args.lr)
        print('')
        print(optimizer)

        # -- Initialise Scheduler
        print('')
        print('Initialising Scheduler...')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

        # -- Load and Initialise Saved Model to Resume Training
        if args.resume_training == 'on':
            checkpoint = torch.load(args.resume_training_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            chkpoint_epoch = checkpoint['epoch']
            chkpoint_loss = checkpoint['loss']
            net.train()
            print('')
            print('Resuming training from epoch', chkpoint_epoch, 'with training loss', chkpoint_loss)

        # -- Initialise SummaryWriter
        if args.resume_training == 'on':
            summarywriter_name = args.resume_training_sumwri_dir
            # - Reference to determine total step number - Author: schaal (Leon), Website: https://discuss.pytorch.org/t/current-step-from-optimizer/19370/2
            chkpoint_step = optimizer.state[optimizer.param_groups[0]['params'][-1]]['step']
            print('Resuming from global step:', chkpoint_step)
            # TODO Find a way to purge both step and epoch
            writer = SummaryWriter(log_dir=summarywriter_name, purge_step=chkpoint_step+1)     # Need to use +1 in purge_step otherwise will lose last data point
        elif args.resume_training == 'off':
            summarywriter_name = str('_' + str(args.network_architecture) + '_' + str(args.epochs) + 'epochs' +
                                     '_patches' + str(args.patches_mode) + '_batchsize' + str(batch_size) +
                                     '_std' + args.checkpoint_std + 'mean' + args.checkpoint_mean
                                     )
            writer = SummaryWriter(comment=summarywriter_name)

        # -- Set Starting Epoch
        if args.resume_training == 'on':
            start_epoch = chkpoint_epoch
        elif args.resume_training == 'off':
            start_epoch = 0

        # -- Set When To Save Checkpoints
        # temp_model_list = range(0, int(args.epochs), int(math.ceil(args.epochs / 10)))     # save model at approx every 10%
        temp_model_list = range(0, int(args.epochs), args.checkpoints_epochs)     # save model every args.checkpoints_epochs

        # -- Start Training the CNN Denoiser
        print('')
        print('Start Training...')
        t1 = time.time()
        for epoch in range(start_epoch, args.epochs):

            running_loss = 0.0

            for i, train_data in enumerate(train_loader, 0):

                # -- Use correct fly mode given data creation mode
                if args.gen_noisy_data_fly_mode == 'ISBI_2022_Paper':

                    # -- Use correct fly mode given network architecture
                    if args.network_architecture == 'DRUNet_Single_Level':

                        # print('Applying noise on the fly with for ISBI_2022_Paper - DRUNet_Single_Level...')

                        # -- Get Data and Send to Device
                        x_noisy_inputs, x_gt = train_data[0].to(device), train_data[1].to(device)

                        # -- Generate Noisy Data: Add Noise to x_noisy_inputs on the device
                        x_noisy_inputs = generate_noisy_data_fly(x_noisy_inputs, settings=args_main_train_settings).to(device)

                    elif args.network_architecture == 'DRUNet_Multi_Level':

                        # print('Applying noise on the fly with ISBI_2022_Paper - DRUNet_Multi_Level...')

                        # -- Get Data and Send to Device
                        x_noisy_inputs, x_gt = train_data[0].to(device), train_data[1].to(device)

                        # -- Generate Noisy Data

                        # - Generate numpy noise_sigma_orig (random std value) and numpy noise_map (noise map using
                        #   random std value)
                        # Convert noise_map to torch so it can be concatenated along channels of x_noisy_inputs
                        noise_sigma_orig, noise_map = generate_noise_map_fly(input_data=x_noisy_inputs, settings=args_main_train_settings)
                        noise_map = torch.from_numpy(noise_map).to(device)

                        # - Set temporary values which are used to generate noisy data
                        args_main_train_settings.noise_gauss_blind_mean_tmp_val = args.gauss_blind_mean
                        args_main_train_settings.noise_gauss_blind_std_tmp_val = noise_sigma_orig

                        # - Generate noisy data by adding noise and send to device
                        x_noisy_inputs = generate_noisy_data_fly_blind(x_noisy_inputs, settings=args_main_train_settings).to(device)

                        # - Concatenate noise_map to x_noisy_inputs along channels
                        x_noisy_inputs = torch.cat((x_noisy_inputs, noise_map), dim=1)

                # -- Zero Parameter Gradients
                optimizer.zero_grad()

                # -- Pass Noisy Images through the Net
                outputs = net(x_noisy_inputs)

                # -- Calculate Error and Backpropagate Error Through Network
                loss = criterion(outputs, x_gt)
                loss.backward()

                # -- Update Parameters
                optimizer.step()

                # -- Print Training Loss
                running_loss += loss.item()
                print_train_loss_num = 1                # print loss after this many steps/iterations
                if i % print_train_loss_num == 0:
                    print('[Epoch: %d, Step: %d] Loss: %.8f' % (epoch + 1, i + 1, running_loss / print_train_loss_num))      # Do division to get average loss for print_train_loss_num iterations/steps
                    running_loss = 0.0                  # reset to 0 for new x number of step's loss and to take average

            # -- Update Learning Rate If Necessary
            scheduler.step()

            # -- Write to SummaryWriter After Every Epoch
            writer.add_scalar('1.training_loss_epoch', loss.item(), epoch+1)      # Starts writing at epoch 1 and not epoch 0

            # -- Save Checkpoints
            # Note: Run if epoch is in temp_model_list or epoch is last epoch
            if (epoch+1 in temp_model_list) or (epoch+1 == args.epochs):
                # - Create checkpoint path and name
                current_datetime = datetime.datetime.now()
                checkpoint_datetime = str(current_datetime.date()) + '_' + str(current_datetime.strftime('%H-%M-%S'))
                checkpoint_name = str('checkpoint_' + str(checkpoint_datetime) +
                                      '_' + str(args.network_architecture) + '_epoch' + str(epoch + 1) +
                                      '_patches' + str(args.patches_mode) + '_batchsize' + str(batch_size) +
                                      '_std' + args.checkpoint_std + 'mean' + args.checkpoint_mean
                                      + '.pt')
                temp_model_path = os.path.join(args.checkpoints_dir, checkpoint_name)
                # - Save checkpoint
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item()
                            }, temp_model_path)
                print('Checkpoint Saved')

        t2 = time.time()
        print('Training Time (Seconds): ', t2-t1, 's')
        print('Training Time (Mins): ', ((t2-t1) / 60), 'mins')
        print('Training Time (Hours): ', (((t2-t1) / 60) / 60), 'hrs')
        print('Training Time (Days): ', ((((t2-t1) / 60) / 60) / 24), 'days')
        print('Finished Training')

        # -- Save Final Model
        # - Create final model path and name
        current_datetime = datetime.datetime.now()
        final_model_datetime = str(current_datetime.date()) + '_' + str(current_datetime.strftime('%H-%M-%S'))
        final_model_name = str('final_model_' + str(final_model_datetime) +
                               '_' + str(args.network_architecture) + '_epoch' + str(epoch + 1) +
                               '_patches' + str(args.patches_mode) + '_batchsize' + str(batch_size) +
                               '_std' + args.checkpoint_std + 'mean' + args.checkpoint_mean
                               + '.pt')
        final_model_path = os.path.join(args.final_model_dir, final_model_name)
        torch.save({'epoch': epoch+1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                    }, final_model_path)
        print('Final Model Output')


    train_init_settings()
