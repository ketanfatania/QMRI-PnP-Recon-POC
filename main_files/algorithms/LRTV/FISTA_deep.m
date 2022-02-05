function [x, ind, density] = FISTA_deepz(data, param)
%
% NOTE:%%%%%%%%%%%%  THIS CODE IS COSTOMIZED FOR ISMRM 19 SUBMISSION and is a CASCADE approach not FULLY ITERATIVE
%
%  solves     min_x {  0.5*|y-F.forward(x)|_2^2 + param.K |Wx|_(1,2)   + MRFNET for NMR parmeter estimation   }
%
%  Inputs:
%
%         y           % observed data
%         F           % pointer to measurement matrix structure
%         D.macth     % deep matching
%         K           % wavelet (group) soft thresholding
%         step        % FISTA stepsize
%         tol         % premature stopping criteria |err_old-err_new| < tol
%         max_iter    % maximum number of iterations
%         N,M,L       % dimension of the image sequence X
%
%   Outputs:
%
%             x       % output image sequence
%             ind     % spatial map of dictionary indices. Can be used to
%                     % calculate the T1, T2, df maps from the look_up table
%             density % density map for the image
%
% (c) Mohammad Golbabaee (m.golbabaee@bath.ac.uk)

%dwtmode('per');

% --- Initializations
N = data.N; M = data.M; L = data.L; 
D = data.D; % nn matching
y = data.y; F = data.F;
T=size(y); T=T(end);


tol = param.tol;
max_iter = param.iter;
if ~isfield(param,'backtrack'), param.backtrack = 1; end


x = zeros(N,M,L);
% FISTA for
t = 1;
x2_prev = x;
obj_prev = 0;
step = param.step;

paramtv.useGPU=param.usegpu;
paramtv.verbose = 0;
%% FISTA for Wavelet L1
for i = 1:max_iter
    
    Fx = F.forward(x);
    
    err = Fx - y;
    
    grad1 =   F.adjoint(err) ;
    cvxobj = 1/2*norm(err(:))^2 ;
    %[val] = wavethresh_group(x, []);
    
    val = norm_tv([reshape(real(x),N,[]); reshape(imag(x),N,[])] );
    
    %---------backtracking line search
    done = 0;
    while ~done
        x2    =   x - grad1 * step;
        %[~,x2]=  wavethresh_group(x2, step*param.K);
        
        if param.K>0
        x2  = [reshape(real(x2),N,[]); reshape(imag(x2),N,[])];
        [x2,~] = prox_tv(x2, step*param.K, paramtv); 
        x2 = x2(1:N,:) +1j*x2(N+1:end,:);
        x2 = reshape(x2,N,M,[]);
        end
        
        %tmp = 1/2*(norm( reshape(F.forward(x2) - y,[],1))^2   +   rho* norm(x2(:)-z(:)+u(:))^2 );
        
        if ~ param.backtrack
            done = 1;
        else
            tmp = 1/2* norm( reshape(F.forward(x2) - y,[],1))^2;
            if (tmp > cvxobj +  real( grad1(:)'*(x2(:)-x(:)))  + 1/(2*step) * norm(x2(:)-x(:))^2)
                disp('reducing stepsize...');
                step = step/2;
            else
                done = 1;
            end
        end
    end
    %----------------------
    x     =   x2 + (t-1)/(t+2)* (x2-x2_prev);
    
    x2_prev = x2;
    t = t +1;
    
    obj = cvxobj + param.K*val;
    fprintf('=== Iter=%i, Obj_FISTA: |y-Ax|^2  + la|W|_1,2 =%e\n', i,obj);
    
    if abs(obj-obj_prev)/obj <tol; break;end
    obj_prev = obj;
end


%% projection step
%[density, ind ] = D.match(x);
density=0; ind = 0;

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
fprintf('|y-F.forward(x)|/|y|= %e\n',...
    norm(err(:))/norm(y(:)) );
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
end







function [val,x] = wavethresh(x,K)
%disp('Wavelet soft thresholding');
L = size(x,3);
J = floor(log2(size(x,1)));
cr=[];ci=[];
sr=[];si=[];
for i =1:L
    [cr(i,:),sr(:,:,i)] = wavedec2(real(x(:,:,i)),J,'db4');
    [ci(i,:),si(:,:,i)] = wavedec2(imag(x(:,:,i)),J,'db4');
end
val = sum(abs([cr(:);ci(:)]));

if nargout > 1
    disp('Wavelet soft thresholding');
    cr = wthresh(cr,'s',K);
    ci = wthresh(ci,'s',K);
    
    for i =1:L
        x(:,:,i)= waverec2(cr(i,:),sr(:,:,i),'db4')  + 1j*waverec2(ci(i,:),si(:,:,i),'db4') ;
    end
end
end


function [val,x] = wavethresh_group(x,K)

L = size(x,3);
J = floor(log2(size(x,1)));
cr=[];ci=[];
sr=[];si=[];
for i =1:L
    [cr(i,:),sr(:,:,i)] = wavedec2(real(x(:,:,i)),J,'db4');
    [ci(i,:),si(:,:,i)] = wavedec2(imag(x(:,:,i)),J,'db4');
end
tmp = sqrt( sum(abs([cr;ci]).^2,1) );
val = sum(tmp);

if nargout > 1
    disp('Wavelet soft thresholding');
    tmp = max(tmp -K,0)./tmp;
    cr = bsxfun(@times, tmp,cr);
    ci = bsxfun(@times, tmp,ci);
    
    for i =1:L
        x(:,:,i)= waverec2(cr(i,:),sr(:,:,i),'db4')  + 1j*waverec2(ci(i,:),si(:,:,i),'db2') ;
    end
end
end