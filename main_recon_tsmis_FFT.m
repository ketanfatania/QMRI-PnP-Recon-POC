% Authors:          Mo Golbabaee, Ketan Fatania
% Email:            m.golbabaee@bath.ac.uk, kf432@bath.ac.uk
% Affiliation:      University of Bath
% GitHub:           https://github.com/ketanfatania/QMRI-PnP-Recon-POC
% Date:             --/09/2020 to --/02/2022

%{
@inproceedings{ref:fatania2022,
	author = {Fatania, Ketan and Pirkl, Carolin M. and Menzel, Marion I. and Hall, Peter and Golbabaee, Mohammad},
	booktitle = {2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
	title = {A Plug-and-Play Approach to Multiparametric Quantitative MRI: Image Reconstruction using Pre-Trained Deep Denoisers},
	code = {https://github.com/ketanfatania/QMRI-PnP-Recon-POC},
	year = {2022}
    }
%}

% -- Dataset used
% The QMaps were provided by GE Healthcare and is not available to be 
% shared for this demo. The "ground-truth" QMaps were computed from long
% FISP aqcuisitions with T=1000 timepoints using the LRTV reconstruction
% algorithm.


%%

clc;
clear;
close all;
addpath(genpath('./'));


%% Settings: General
% For further settings, please see sections "Load PyTorch TSMI 
% Denoiser...", "TSMI Reconstruction" and "Dictionary Matching"

% -- Test data selection
vol = 8;                                % Subject/Volunteer to use. 8 was held out during training
slice = 10;                             % TSMI brain slice to subsample and reconstruct

% -- Acquisition type and truncation length
% Note: cut = 0 is the untruncated fisp sequence
% cut=0 (T=1000), cut=1 (T=500), cut=2 (T=300), cut=3 (T=200), cut=4 (T=100)
scan_type = 'fisp';                     % 'fisp'
cut = 3;                                % 0, 1, 2, 3 or 4

% -- Subsampling pattern and corresponding compression ratio
subsampling_pattern = 'Spiral';         % 'Spiral' or 'EPI'
spiral_sampling_curve = 771;            % number of samples taken per timeframe on the spiral
epi_sampling_rate = 1/65;               % ~ 1/(pixel distance) between two epi readout lines

% -- Add Gaussian noise to Y measurements
measurements_type = 'noisy';            % 'noisy' or 'clean'
measurements_noise = 30;                % 30

% -- TSMI reconstruction method
recon_method = 'PnP_ADMM';              % 'SVD_MRF', 'LRTV', 'PnP_ADMM'


% -- Convert sampling rate to string to name files later
if strcmp(subsampling_pattern, 'Spiral')
    sampling_rate_str = num2str(spiral_sampling_curve);
elseif strcmp(subsampling_pattern, 'EPI')
    sampling_rate_str = num2str(epi_sampling_rate);
end


%% Settings: Denoiser

% -- Denoiser type and noise level
% 'single_level' = the noise_map_std value is ignored and
%                  denoising will be performed with the single noise 
%                  denoiser.
% 'multi_level' = the noise_map_std value will be used to create a noise 
%                 map for channel 11 of the multi-level noise denoiser.
denoiser_type = 'single_level';         % 'single_level' or 'multi_level'
noise_map_std = 0.01;                   % denoiser trained from 0.0001std to 1std

% -- Denoiser directory
denoiser_dir = ['./onnx_models/real_',scan_type,'_cut',num2str(cut),'_onnx_models/'];

% -- Denoiser filenames
single_level_denoiser_filename = 'pytorch_model_drunet_10ch_500epochs.onnx';
multi_level_denoiser_filename = 'pytorch_model_drunet_11ch_500epochs.onnx';

% -- Create denoiser path
if ~isfolder(denoiser_dir)
    mkdir(denoiser_dir)
end
switch denoiser_type
    case 'single_level'
        denoiser_path = [denoiser_dir, single_level_denoiser_filename];
    case 'multi_level'
        denoiser_path = [denoiser_dir, multi_level_denoiser_filename];
end


%% Settings: Save or Load Y Measurements
% For reproducible results and fair comparison across reconstruction
% methods when adding noise to k-space measurements.
% Instructions: First run with 'save', then modify
%               'load_measurements_filename' to newly saved file and  
%               finally run with 'load'.
save_load_measurements = 'off';            % 'save', 'load', 'off'

% -- Measurements dir
measurements_dir = ['./saved_y_measurements/real_',scan_type,'_cut',num2str(cut),'_measurements/'];

% -- Measurements filenames
save_measurements_filename = ['y_measurements_vol',num2str(vol),'slice',num2str(slice),'_',lower(subsampling_pattern),'_',sampling_rate_str,'.mat'];
load_measurements_filename = 'y_measurements_vol8slice10_spiral_771.mat';

% -- Create measurements path
if ~isfolder(measurements_dir)
    mkdir(measurements_dir)
end
save_measurements_path = [measurements_dir, save_measurements_filename];
load_measurements_path = [measurements_dir, save_measurements_filename];


%% Load Dictionary

disp('Loading dictionary...')

% -- Load the dictionary that corresponds to the selected acquisition type
switch scan_type
    case 'fisp'
        dict_dir = ['./dictionaries/real_fisp_cut',num2str(cut),'_dict/SVD_dict_FISP_cut',num2str(cut),'.mat'];
        load(dict_dir);
        V = real(dict.V);
end


%% Load PyTorch TSMI Denoiser Exported to ONNX Format and Create Noise Map

disp('Loading ONNX model...')

% -- Import ONNX model as DAGNetwork and then convert from DAGNetwork to LayerGraph
Net = importONNXNetwork(denoiser_path, 'OutputLayerType', 'regression');
lgraph = layerGraph(Net);           

% -- Create custom input layer and set input dimensions
% Normalization steps are performed during PnP-ADMM iterations
switch denoiser_type
    case 'single_level'
        larray = imageInputLayer([224 224 10],'Name','In', 'Normalization','none');
    case 'multi_level'
        larray = imageInputLayer([224 224 11],'Name','In', 'Normalization','none');
end

% -- Replace input layer in lgraph and build network
lgraph = replaceLayer(lgraph,'Input_input',larray);
Net = assembleNetwork(lgraph);
% plot(Net)                                     % To visualise the network

% -- Modify Matlab internal code to enable use of denoiser
% Note: denoiseImage.m was modified to create denoiseImage_PnP_ADMM.m
% onnx_dagnetwork: true - allows use of onnx model as dagnetwork 
%                  false - uses the original Matlab code
% residual_noise: true - uses original Matlab code which removes 
%                        CNN output from the original input
%                 false - returns CNN output 
onnx_dagnetwork = true;                                   
residual_noise = false;                                     
param.net = @(x) denoiseImage_PnP_ADMM(x, Net, onnx_dagnetwork, residual_noise);      

% -- Build noise map if using multi-level noise denoiser
param.denoiser_type = denoiser_type;
if strcmp(param.denoiser_type, 'multi_level')
    disp('Building noise map...')
    param.noise_map = build_noise_map(noise_map_std, 224, 224);
end


%% Load Ground-Truth QMaps, Crop QMaps and Create Foreground Mask

disp('Loading QMaps...')

% -- Load ground-truth quantitative tissue maps
MRFmaps_dir = ['./datasets/gt_qmaps/qmap_gt_vol', num2str(vol), '.mat'];
load(MRFmaps_dir);                          % Batch x C x W x H
[~,~,N,M] = size(qmap);
L = size(dict.V,2);
qmap = qmap(slice,:,:,:);
qmap = permute(qmap,[1,3,4,2]);
qmap = reshape(qmap,[],3);
qmap0 = squeeze(reshape(qmap,[],N,M,3));

% -- Crop qmaps
qmap0 = qmap0((4:227),(4:227),:);
[N,M,~] = size(qmap0);

% -- Create foreground mask to account for artefacts created due to air
foreground_mask =  getmask_fromPD( qmap0(:,:,3), 0.15);     % This mask will be used during calculation of metrics.
ind_foreground_mask = find(foreground_mask>0);              % Mask indices

clear X; clear qmap;


%% Load TSMIs and Crop TSMIs for U-Net

disp('Loading TSMIs...')

% -- Load 10ch TSMI slice for testing
switch scan_type
    case 'fisp'
        tsmi_dir = ['./datasets/synth_tsmis/real_fisp_cut', num2str(cut), '_tsmis/testing_data/vol', num2str(vol), 's', num2str(slice),'.mat'];
        load(tsmi_dir);
        X0 = X;
end

% -- Crop TSMI slice from 230x230x10 to 224x224x10
X0 = X0((4:227),(4:227), :);


%% Build Gridded FFT Subsampling Operators
% FAST implementation of subsampling operators using sparse matrix product

disp(['Building ', subsampling_pattern, ' sampling operators...'])

switch subsampling_pattern
    case 'Spiral'
        [P] = setup_subsampling_spiralgrided(N,M,spiral_sampling_curve,V); 
    case 'EPI'
        [P] = setup_subsampling_epi(N,M,epi_sampling_rate,V);
end

% -- Create forward and adjoint subsampling operators
F.forward = @(x) P.for(reshape(fft2(x),[],1))/sqrt(N*M);
F.adjoint = @(x) (ifft2(reshape(P.adj(x),N,M,[]))*sqrt(N*M));


%% Subsample TSMIs and Add Noise to Y Measurements

disp(['Subsampling TSMIs using ', subsampling_pattern, ' pattern...'])

% -- Subsample TSMI
Y = F.forward(double(X0));

% -- Add noise
switch measurements_type
    case 'noisy'
        disp('Adding noise to measurements...')
        Y = awgn(Y, measurements_noise, 'measured');
    case 'clean'
        disp('No noise added to measurements...')
end


%% Save and Load Y measurements

switch save_load_measurements
    case 'save'
        save(save_measurements_path, 'Y');
        disp('Measurements saved!')
        return
    case 'load'
        loaded_measurements = load(load_measurements_path);
        Y = loaded_measurements.Y;
        disp(['Y measurements loaded from ', load_measurements_path]);
end


%% TSMI Reconstruction

disp(['TSMI reconstruction using ', recon_method, '...'])
disp(' ')

switch recon_method
    
    case 'SVD_MRF'                          % SVD-MRF: McGivney et al. 2014
        out.X = F.adjoint(double(Y));
    
    case 'LRTV'                             % LRTV: Golbabaee et al. 2021
        param.K = 4e-5;
        param.iter = 200;
        param.step = numel(X0)/numel(Y);
        param.tol = 1e-4;
        param.backtrack = 1;
        param.usegpu = 0;                   
        paramTV.TVop = TV_operator('2D',param.usegpu);
        data.N=M;  data.M=M; data.L=L; data.y = double(Y); data.F = F; data.D = [];
        [out.X] = FISTA_deep(data, param);
        
    case 'PnP_ADMM'                         % PnP-ADMM Iterations: Venkatakrishnan et al. 2013, Ahmad et al. 2020
        param.eta = 20;
        param.sigma_squared = 1;
        param.gamma = param.sigma_squared/param.eta;    % Paper's gamma parameter. 1/20 = 0.05
        param.iter = 100;                               % Number of PnP-ADMM iterations
        param.cg_tol = 1e-4;                            % Conjugate-Gradient tolerance
        param.F = F;                                    % F contains forward and adjoint operators
        param.gt_tsmi = X0;                             % GT TSMI for metrics
        param.X0 = F.adjoint(Y);                        % Reconstruct SVD-MRF image
        [out.X] = PnP_ADMM(double(Y), param);           % Perform PnP-ADMM

end


%% Dictionary Matching

disp(' ')
disp('Dictionary matching for quantitative inference...')

% -- Settings for dictionary matching
par.f.qout = 1;
par.f.pdout = 1;
par.f.mtout = 0;
par.f.Xout = 0;
par.f.dmout=0;
par.f.Yout = 0;
par.fp.blockSize = 1e9;
par.f.verbose = 1;

% -- Get reconstructed TSMIs
X = out.X;

% -- Perform dictionary matching for quantitative inference
out = mrf_dtm_cpu(dict, out, par);
qmap = cat(3,out.qmap,out.pd);

% -- Resets GPU if using GPU
% a = gpuDevice(1);
% wait(a);
% reset(a);


%% Prepare QMaps for Calculation of Metrics and Visualisation

qmap        =   double(qmap);
qmap0       =   double(qmap0);

t1          =    qmap(:,:,1).*foreground_mask;
t1_ref      =    qmap0(:,:,1).*foreground_mask;
t2          =    qmap(:,:,2).*foreground_mask;
t2_ref      =    qmap0(:,:,2).*foreground_mask;
pd          =    qmap(:,:,3).*foreground_mask;
pd          =    abs(pd)/max(abs(pd(:)));           % Compare absolute of pd & scale pd to unity (a.u.)
pd_ref      =    qmap0(:,:,3).*foreground_mask;
pd_ref      =    abs(pd_ref)/max(abs(pd_ref(:)));   % Compare absolute of pd & scale pd to unity (a.u.)


%% Calculate and Display Metrics

disp(' ')
disp('Displaying metrics...')

% -- Calculate MAE for QMaps with foreground mask. 0 is perfect.
t1_mae = mean( abs( t1(ind_foreground_mask) - t1_ref(ind_foreground_mask) ) );
t2_mae = mean( abs( t2(ind_foreground_mask) - t2_ref(ind_foreground_mask) ) );
pd_mae = mean( abs( pd(ind_foreground_mask) - pd_ref(ind_foreground_mask) ) );

% -- Calculate PSNR for QMaps without foreground mask. Inf is perfect.
t1_psnr = psnr(t1, t1_ref);
t2_psnr = psnr(t2, t2_ref);
pd_psnr = psnr(pd, pd_ref);

% -- Calculate SSIM for QMaps without foreground mask. 1 is perfect.
t1_ssim = ssim(t1, t1_ref);
t2_ssim = ssim(t2, t2_ref);
pd_ssim = ssim(pd, pd_ref);

% -- Calculate mean PSNR for TSMIs without foreground mask. Inf is perfect.
num_channels = size(X0, 3);
psnr_tsmi_ch_results = zeros(num_channels, 1, 'double');
for psnr_tsmi_ch = 1 : num_channels
    psnr_tsmi_ch_results(psnr_tsmi_ch) = psnr(abs(X(:,:,psnr_tsmi_ch)), abs(X0(:,:,psnr_tsmi_ch)));
end
psnr_tsmi_ch_mean = mean(psnr_tsmi_ch_results);

% -- Calculate mean SSIM for TSMIs without foreground mask. 1 is perfect.
ssim_tsmi_ch_results = zeros(num_channels, 1, 'double');
for ssim_tsmi_ch = 1 : num_channels
    ssim_tsmi_ch_results(ssim_tsmi_ch) = ssim(abs(X(:,:,ssim_tsmi_ch)), abs(X0(:,:,ssim_tsmi_ch)));
end
ssim_tsmi_ch_mean = mean(ssim_tsmi_ch_results);

% -- Display metrics
disp(' ')
fprintf(' tsmi_mean_psnr=%.6f\n tsmi_mean_ssim=%.6f\n' , psnr_tsmi_ch_mean, ssim_tsmi_ch_mean);
disp(' ')
fprintf(' t1_mae=%.6f\n t1_psnr=%.6f\n t1_ssim=%.6f\n', t1_mae, t1_psnr, t1_ssim);
disp(' ')
fprintf(' t2_mae=%.6f\n t2_psnr=%.6f\n t2_ssim=%.6f\n', t2_mae, t2_psnr, t2_ssim);
disp(' ')
fprintf(' pd_mae=%.6f\n pd_psnr=%.6f\n pd_ssim=%.6f\n', pd_mae, pd_psnr, pd_ssim);


%% Visualisation of Quantitative Tissue Maps

% -- Quantitative Tissue Maps: Ground Truth
figure(1)
subplot(131);imagesc(t1_ref); caxis([0,3]); axis off; colormap jet; title('GT T1');colorbar('southoutside')
subplot(132);imagesc(t2_ref); caxis([0,.3]);axis off; colormap jet; title('GT T2');colorbar('southoutside')
subplot(133);imagesc(pd_ref); caxis([0,1]); axis off;colormap jet; title('GT PD');colorbar('southoutside')

% -- Quantitative Tissue Maps: Inferred 
figure(2)
subplot(131);imagesc(t1); caxis([0,3]); axis off; colormap jet; title('Recon T1');colorbar('southoutside')
subplot(132);imagesc(t2); caxis([0,.3]);axis off; colormap jet; title('Recon T2');colorbar('southoutside')
subplot(133);imagesc(pd); caxis([0,1]); axis off;colormap jet; title('Recon PD');colorbar('southoutside')

% -- Quantitative Tissue Maps: Error Maps
figure(3) 
subplot(131);imagesc(t1-t1_ref); caxis([-.2,.2]); axis off;colormap bone; title('T1 Error'); colorbar('southoutside')
subplot(132);imagesc(t2-t2_ref); caxis([-0.05,.05]); axis off;colormap bone; title('T2 Error');colorbar('southoutside')
subplot(133);imagesc(pd-pd_ref); caxis([-.1,.1]); axis off;colormap bone; title('PD Error');colorbar('southoutside')


%% Visualisation of TSMIs

% -- TSMIs: Ground Truth
figure(4)
subplot(2,5,1);imagesc(abs(X0(:,:,1))); axis off;colormap bone; title('GT Ch1'); colorbar('southoutside')
subplot(2,5,2);imagesc(abs(X0(:,:,2))); axis off;colormap bone; title('GT Ch2');colorbar('southoutside')
subplot(2,5,3);imagesc(abs(X0(:,:,3))); axis off;colormap bone; title('GT Ch3');colorbar('southoutside')
subplot(2,5,4);imagesc(abs(X0(:,:,4))); axis off;colormap bone; title('GT Ch4'); colorbar('southoutside')
subplot(2,5,5);imagesc(abs(X0(:,:,5))); axis off;colormap bone; title('GT Ch5');colorbar('southoutside')
subplot(2,5,6);imagesc(abs(X0(:,:,6))); axis off;colormap bone; title('GT Ch6');colorbar('southoutside')
subplot(2,5,7);imagesc(abs(X0(:,:,7))); axis off;colormap bone; title('GT Ch7'); colorbar('southoutside')
subplot(2,5,8);imagesc(abs(X0(:,:,8))); axis off;colormap bone; title('GT Ch8');colorbar('southoutside')
subplot(2,5,9);imagesc(abs(X0(:,:,9))); axis off;colormap bone; title('GT Ch9');colorbar('southoutside')
subplot(2,5,10);imagesc(abs(X0(:,:,10))); axis off;colormap bone; title('GT Ch10');colorbar('southoutside')

% -- TSMIs: Reconstructed
figure(5) 
subplot(2,5,1);imagesc(abs(X(:,:,1))); axis off;colormap bone; title('Recon Ch1'); colorbar('southoutside')
subplot(2,5,2);imagesc(abs(X(:,:,2))); axis off;colormap bone; title('Recon Ch2');colorbar('southoutside')
subplot(2,5,3);imagesc(abs(X(:,:,3))); axis off;colormap bone; title('Recon Ch3');colorbar('southoutside')
subplot(2,5,4);imagesc(abs(X(:,:,4))); axis off;colormap bone; title('Recon Ch4'); colorbar('southoutside')
subplot(2,5,5);imagesc(abs(X(:,:,5))); axis off;colormap bone; title('Recon Ch5');colorbar('southoutside')
subplot(2,5,6);imagesc(abs(X(:,:,6))); axis off;colormap bone; title('Recon Ch6');colorbar('southoutside')
subplot(2,5,7);imagesc(abs(X(:,:,7))); axis off;colormap bone; title('Recon Ch7'); colorbar('southoutside')
subplot(2,5,8);imagesc(abs(X(:,:,8))); axis off;colormap bone; title('Recon Ch8');colorbar('southoutside')
subplot(2,5,9);imagesc(abs(X(:,:,9))); axis off;colormap bone; title('Recon Ch9');colorbar('southoutside')
subplot(2,5,10);imagesc(abs(X(:,:,10))); axis off;colormap bone; title('Recon Ch10');colorbar('southoutside')
