function out = mrf_dtm_cpu(dict,data,par)
%% Dictionary template matching for MRF
% =========================================================================
% out = mrf_dtm(dict,data,par)
% %
%   Input
%         
%       dict
%           .D          [K entries, L elements]   dictionary
%           .normD      [K entries, 1]             norm of dictionary
%           .lut        [K entries, L elements]  
%           .V          [K entries, V singular vectors] temporal subspace of dict (optional) 
%       data
%           .X          [Nx,Ny,(Nz), T timepoints]   Reconstructed data 
%           .mask       [Nx,Ny,(Nz)]                 Fitting mask, optional
%       par
%
%   Output
%       out.X           [Nx,Ny,(Nz), T timepoints]     fit of data Xf=norm(Xi)/norm(D)*D
%       out.qmap        [Nx,Ny,(Nz), Q maps]           estimated biological parameters
%       out.mt          [Nx,Ny,(Nz)]                   matching result (voxel-atom correlation)
%       out.dm          [Nx,Ny,(Nz)]                   selected dictionary entries in L
%       out.pd          [Nx,Ny,(Nz)]                   proton density  
%
%   Function
%       Finds the highest correlated atom in the dictionary to the data
%       by projection max <Di,x>
% =========================================================================
% Log
% v6.1: 06.12.17 - Dictionary match as standalone function, removed
% previous log. Log in mrf_dtm_fit
% =========================================================================
% (c) 2015 - 2019 P. Gomez
% =========================================================================

%% Get variables

Q = size(dict.lut, 2);
datadims = size(data.X); %[par.ind.mtx_reco par.ind.mtx_reco size(dict.V, 2)];

if length(datadims)>=4
    [ix,iy,iz,T] = size(data.X);
else
    iz = 1;
    [ix,iy,T] = size(data.X);
end
N = ix*iy*iz;

%% Reshape and mask data
x = reshape(data.X,[N,T]);
if 1%~isfield(data,'mask')
    data.mask = ones(datadims(1:end-1)) > 0;
end
x = single(x(data.mask>0,:));

%% Get dictionary
% if (par.f.compress && (size(dict.D,2) > par.ind.Tv)) || T < size(dict.D,2)
%     dict.D = dict.D*dict.V(:,1:par.ind.Tv); 
% end
% D = dict.D;
% if isfield(dict,'normD'); normD = dict.normD; end    
% if ~isfield(dict,'lut')
%     error('dict.lut required for parameter estimation'); 
% end
% if ~exist('normD','var')
%     warning('normD not found, normalizing dictionary \n');
%     normD = sqrt(sum((abs(dict.D_nn)).^2,2));
%     D = bsxfun(@rdivide,D,normD);
%     D(isnan(D))=0;
% end

%% DTM Fit

blockSize = min(max(floor(par.fp.blockSize/size(dict.D,1)),1),size(x,1));
 if isempty(blockSize)
     blockSize=size(x,1);
 end
 iter = ceil(size(x,1)/blockSize);
mt = zeros(size(x,1),1);
dm = zeros(size(x,1),1);
pd = zeros(size(x,1),1);
X = zeros(size(x));
if par.f.verbose; fprintf('Matching data \n'); end
for i = 1:iter
    if par.f.verbose; fprintf(['Matching block ', num2str(i) '/' num2str(iter), '\n']); end
    if i<iter
        cind = (i-1)*blockSize+1:i*blockSize;    
    else
        cind = (i-1)*blockSize+1:size(x,1); 
    end
        ip=dict.D*ctranspose(x(cind,:)); 
        [mt(cind),dm(cind)] = max(abs(ip),[],1);
        for ic = 1:length(cind)
            pd(cind(ic)) = ip(dm(cind(ic),1),ic);
            X(cind(ic),:) = pd(cind(ic)).*dict.D(dm(cind(ic),1),:); %re-scale X as in Davies et al.
            pd(cind(ic)) = pd(cind(ic))./dict.normD(dm(cind(ic),1)); %get PD after signal scaling. 
        end       
end

%%
% mt = zeros(size(x,1),1,'gpuArray');
% dm = zeros(size(x,1),1,'gpuArray');
% pd = zeros(size(x,1),1,'gpuArray');
% X = zeros(size(x),'gpuArray');
% dict.D=gpuArray(dict.D);
% x=gpuArray(x);
% 
% if par.f.verbose; fprintf('Matching data \n'); end
% for i = 1:iter
%     if par.f.verbose; fprintf(['Matching block ', num2str(i) '/' num2str(iter), '\n']); end
%     if i<iter
%         cind = (i-1)*blockSize+1:i*blockSize;    
%     else
%         cind = (i-1)*blockSize+1:size(x,1); 
%     end
%         ip=dict.D*ctranspose(x(cind,:)); 
%         [mt(cind),dm(cind)] = max(abs(ip),[],1);
%         for ic = 1:length(cind)
%             pd(cind(ic)) = ip(dm(cind(ic),1),ic);
%             X(cind(ic),:) = pd(cind(ic)).*dict.D(dm(cind(ic),1),:); %re-scale X as in Davies et al.
%             pd(cind(ic)) = pd(cind(ic))./dict.normD(dm(cind(ic),1)); %get PD after signal scaling. 
%         end       
% end

clear match;
clear dictentry;

%% Out
if par.f.Xout
    Xout = zeros(N,T,'single');
    Xout(data.mask>0,:) = single(X);
    out.Xfit = reshape(Xout,[datadims(1:end-1),T]);
    out.X = data.X;
end

if par.f.qout
    qmap = zeros(N,Q,'single');
    qmap(data.mask>0,:) = dict.lut(dm,:);
    qmap(isnan(qmap)) = 0; %remove possible NaN from infeasible search window
    out.qmap = reshape(qmap,[datadims(1:end-1),Q]);
    out.mask = data.mask;
end

if par.f.pdout
    pdfit = single(zeros(N,1));
    pdfit(data.mask>0,:) = single(pd);
    out.pd =  reshape(pdfit,[datadims(1:end-1),1]);
end

if par.f.mtout
    mfit = single(zeros(N,1));
    mfit(data.mask>0,:) = single(mt);
    out.mt =  reshape(mfit,[datadims(1:end-1),1]);
end

if par.f.dmout
    dmfit = single(zeros(N,1));
    dmfit(data.mask>0,:) = single(dm);
    out.dm =  reshape(dmfit,[datadims(1:end-1),1]);
end

if par.f.Yout && isfield(data,'Y')  
   out.Y = data.Y; 
end

end