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

%%

clc;
clear;
close all;


%% Settings

% -- TSMIs output data type
mode = 'real';                  % 'real' or 'complex'

% -- Acquisition type and truncation length
% Note: cut = 0 is the untruncated fisp sequence
% cut=0 (T=1000), cut=1 (T=500), cut=2 (T=300), cut=3 (T=200), cut=4 (T=100)
dico = 'fisp';                  % 'fisp'
cut = 3;                        % 0, 1, 2, 3 or 4

% -- Number of Volunteers/Subjects for Training and Testing Data
% Note: For ISBI 2022 paper results, the first 7 subjects were for training 
% and subject 8 was used for testing.
train_test_split = 7;           % The final subject used for training data before testing data is created

% -- Path to QMaps
MRFmaps_dir = ['./datasets/gt_qmaps/'];

% -- Output folder
out_root = ['./datasets/synth_tsmis/', mode, '_', dico, '_cut', num2str(cut), '_tsmis', filesep];


%% Load dictionary

if strcmp(dico,'fisp')
    dict_dir = ['./dictionaries/', mode, '_fisp_cut', num2str(cut), '_dict/SVD_dict_FISP_cut', num2str(cut), '.mat'];
    load(dict_dir);
end

Mdl = KDTreeSearcher(dict.lut);


%% Construct TSMIs and save them

disp(['Creating ', mode, '-valued TSMIs using ', dico, ' cut', num2str(cut),'...'])
disp(' ')

qmap_file = dir([MRFmaps_dir 'qmap_gt_vol*.mat']);

for v = 1:length(qmap_file)
    
    disp(['Processing ', qmap_file(v).name, '...']);
    
    load([qmap_file(v).folder filesep qmap_file(v).name],'qmap');
     
    if v<= train_test_split
        out_dir = [out_root 'training_data' filesep];
    else
        out_dir = [out_root 'testing_data' filesep];
    end
    
    if ~isfolder(out_dir)
        mkdir(out_dir)
    end
    
    [S,~,N,M] = size(qmap);
    
    for slice =1:S
        X=[];
        qm = qmap(slice,:,:,:);
        
        qm = permute(qm,[1,3,4,2]);
        qm = reshape(qm,[],3);
        I = knnsearch(Mdl,qm(:,1:2),'K',1);
        X = real(dict.D(I,1:10));
        X = bsxfun(@times, X, dict.normD(I));

        if strcmp(mode,'real')
            X = bsxfun(@times, X, abs(qm(:,3)));
            X = reshape(X,N,M,[]); % 10 channel real TSMI
            
            % Align first SVD channel to be positive (phase zero) 
            s= sign(X(:,:,1));
            X = bsxfun(@times, X, s);
            
        else
            X = bsxfun(@times, X, qm(:,3));
            X = cat(3, reshape(real(X),N,M,[]),reshape(imag(X),N,M,[])); % 20 channel complex TSMI
       
        end
                
        out_file = [out_dir 'vol' num2str(v) 's' num2str(slice) '.mat'];
        save(out_file,'X');  % outputs are single.
                
    end
end

disp(' ')
disp('TSMIs have been created and saved')
