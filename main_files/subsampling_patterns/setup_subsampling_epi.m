function [S] = setup_subsampling_epi(N,M,percentage,V)
% This function generates forward and adjoint sub-sampled FFT operators
% corresponding to the Multi-shot Echo Planar Imaging (EPI) MRF aqcuisition.
% Inputs:
%                 sampling: 'deterministic comb' i.e. the Multi-shot EPI
%                 acquisition
%                 N: number of rows
%                 M: number of columns
%                 percentage: sub-sampling rate
%                 L: number of timepoints in MRF sequence 
 
% Outputs: Forward model
%                 H.forward
%                 H.adjoint
%
% (c) Mohammad Golbabaee, 2017

%%

step = round(1/percentage);
no_of_steps = floor(N/step);
nb_meas = no_of_steps*M;
L = size(V,1);
%mask = zeros(nb_meas*L,1);
comb = zeros(N,1); comb(1:step:step*nb_meas/M) = 1;
P = [];
for i = 1:L
    comb = comb([N,1:N-1]);  %cyclic shift
    template = comb*ones(1,M);
    ind = find(template(:)==1);
    tmp = sparse([1:numel(ind)], ind, ones(1,numel(ind)), numel(ind),N*M);
    P = [P;tmp* kron( conj(V(i,:)),speye(N*M))];
end
S.for = @(x) P*x;
S.adj = @(x) P'*x;

end