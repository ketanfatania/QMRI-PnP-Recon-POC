function [K]=setup_subsampling_spiralgrided(N,M,S,V)

% (c) Mohammad Golbabaee (m.golbabaee@bath.ac.uk) 2021

%%

delta = pi/180 * 7.5; % Shift for the spiral between each slice

%% Solution 2: approximation with discretization & FFT
% The code below generate L masks to select samples in the usual way
% ... The nb of measurements in each slice can change because
%     of the discretization ...

% --- Init
L = size(V,1);
t = linspace(0, 2*pi, S); %10000
theta = 8*t;
r = 1.05.^theta;
r = (r-min(r))/(max(r)-min(r));

% --- Generate subspace-compressed, spatiotemporal, sparse sampling matrix P
P = [];
for i = 1:L
    % Get spiral
    cx = r.*cos(theta + (i-1)*delta); cy = r.*sin(theta + (i-1)*delta);
    
    % Approximate on the grid
    cx = round(cx*N/2)+N/2+1; cy = round(cy*N/2)+N/2+1;
    cx = min(cx, N); cy = min(cy, N);
    ind = cx + N*(cy-1);
    temp = zeros(N); 
    temp(ind(:)) = 1; 
    temp=fftshift(temp);
    ind = find(temp==1);
    
    tmp = sparse([1:numel(ind)], ind, ones(1,numel(ind)), numel(ind),N*M);
    P = [P;tmp* kron( conj(V(i,:)),speye(N*M))];
    
end
%save('./P_op__miccaiqti_spiral.mat','P');
K.for = @(x) P*x;
K.adj = @(x) P'*x;
end