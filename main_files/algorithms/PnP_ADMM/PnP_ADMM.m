function [x] = PnP_ADMM(y, param)
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
% -----------------------------------------------------------------
%  Solves:
%
%               min_x {  0.5*||y - Ax||_2^2 + phi(x)   }
%
%  using PnP-ADMM algorithm:
%
%               x_(k) = h( v_(k-1) - u(k-1) )           (Step 1)
%               v_(k) = f( x(k) + u(k-1) )              (Step 2)
%               u_(k) = u_(k-1) + ( x_(k) - v_(k) )     (Step 3)
%
%       where   h(z) = min_x {  ||y − Ax||_2^2  +  gamma*||x − z||_2^2  }
%               z = v_(k-1) - u(k-1)
%
%         and   x = Reconstructed TSMI: Solve for h using CG algorithm         
%               v = Optimisation parameter: Apply denoising to promote priors
%               u = Optimisation parameter: Aggregates previous two steps 
%               f = Denoiser
%               k = Iteration number
%
%  Inputs:
%
%         x                 % x = param.X0 = SVD-MRF solution (subsampled TSMIs)
%         y                 % Compressed measurements (k-space data)
%         F                 % Pointer to forward/adjoint operators
%         r                 % Convergence parameter, gamma = sigma_squared/eta
%
%         net               % Denoising CNN network
%         denoiser_type     % Single-level(10ch) or multi-level(11ch) denoiser 
%         noise_map         % Noise map for multi-level denoiser
%
%         max_iter          % Maximum number of iterations
%         cg_tol            % Conjugate-Gradient tolerance
%         gt_tsmi           % Ground-truth TSMI for metric calculations
%
%   Outputs:
%
%         x                 % Reconstructed TSMI
%
% -----------------------------------------------------------------


%% Initializations

% -- Initialisation of general parameters
max_iter = param.iter;
r = param.gamma;    
F = param.F;
cg_tol = param.cg_tol;
gt_tsmi = param.gt_tsmi;

% -- Initialisation of denoiser parameters
net = param.net;
denoiser_type = param.denoiser_type;
if strcmp(denoiser_type, 'multi_level')
    noise_map = param.noise_map;
end

% -- Initialisation of PnP-ADMM parameters
x = param.X0;
v = x;
uold = 0;

[dim.dy1,dim.dy2,dim.dy3,dim.dy4,dim.dy5,dim.dy6]= size(y);
[dim.dx1,dim.dx2,dim.dx3,dim.dx4,dim.dx5,dim.dx6]= size(x);

% -- Initialisation of the input parameters of CG-solver
opt.F = F;
opt.dim = dim;
opt.r = r;


%% ADMM loop

disp('Performing PnP-ADMM Iterations...');

for i = 1:max_iter
    
    disp(' ')
    disp(['Iteration: ', num2str(i)])

    
    % ************* Step 1 - The Least Squares Solving Step (using CG Algorithm) ***************
    
    % -- Solve h(z) using Conjugate-Gradient algorithm
    x = lsqr(@(z,flag)afun(z, opt, flag), [y(:);(v(:)-uold(:))*sqrt(r)], cg_tol, 100, [], [], x(:));
    x = reshape(x,dim.dx1,dim.dx2,dim.dx3,dim.dx4,dim.dx5,dim.dx6);
    
    % -- Print progress metrics
    normalized_y_measurements_data_fidelity = norm(reshape(y-F.forward(x), 1, [])) / norm(y(:));
    normalized_tsmi_solution_error = norm(reshape(gt_tsmi-x, 1, [])) / norm(gt_tsmi(:));
    fprintf('-- Normalized_Y_Measurements_Data_Fidelity = %e \n', normalized_y_measurements_data_fidelity);
    fprintf('-- Normalized_TSMI_Solution_Error = %e \n', normalized_tsmi_solution_error);  
    
    
    % **************** Step 2 - The Neural Proximal Step ****************
    
    % -- Calculate v
    v = (x + uold);   

    % -- Use only real parts
    v = real(v);
    
    % -- Normalise 0to1
    [v, x_min, x_max, x_range] = norm_zero_to_one(v);
    
    % -- Denoise with or without noise map
    switch denoiser_type 
        
        case 'single_level'
            % disp('Denoising without Noise Map...')
            v = net(v);
            
        case 'multi_level'
            % disp('Denoising with Noise Map...')
            v = cat(3, v, noise_map);
            v = net(v);  
            
    end
        
    % -- Undo Normalisation of 0to1
    v = undo_norm_zero_to_one(v, x_min, x_max, x_range);
    
    
    % ****************** Step 3 - The Aggregation Step *******************
    
    % -- Perform aggregation step
    uold = uold + x - v;
    
end

end


%% Functions

function y = afun(x,opt,flag)

x = double(x);
dim = opt.dim;
F = opt.F;
r = opt.r;

    if strcmp(flag,'notransp') % Compute A*x
        x = reshape(x,dim.dx1,dim.dx2,dim.dx3,dim.dx4,dim.dx5,dim.dx6);
        y = [ reshape(F.forward(x) ,[],1) ; x(:)*sqrt(r)];

    elseif strcmp(flag,'transp') % Compute A'*x
        s = prod([dim.dy1,dim.dy2,dim.dy3,dim.dy4,dim.dy5,dim.dy6]);
        y = F.adjoint( reshape( x(1:s), dim.dy1,dim.dy2,dim.dy3,dim.dy4,dim.dy5,dim.dy6) );
        y = y(:)+ x(s+1:end)*sqrt(r);
        
    end
    
end


function [x_scaled, x_min, x_max, x_range] = norm_zero_to_one(x_unscaled)

% Initialize Variables
x_min = min(x_unscaled(:));
x_max = max(x_unscaled(:));
x_range = x_max - x_min;

% Normalize 0 to 1
x_scaled = ((x_unscaled - x_min) / x_range);

end


function [x_unscaled] = undo_norm_zero_to_one(x_scaled, x_min, x_max, x_range)

% Undo Normalization of 0 to 1
x_unscaled = (x_scaled * x_range) + x_min;

end
