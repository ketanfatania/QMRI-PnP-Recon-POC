function noise_map = build_noise_map(noise_std, rows, cols)

% Author:           Ketan Fatania
% Email:            kf432@bath.ac.uk
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

noise_map = repmat(noise_std, rows, cols);

