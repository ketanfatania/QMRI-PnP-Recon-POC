function mask = getmask_fromPD(PD, thresh)
% A basic foreground masking for MRF by thresholdng the Proton density intensity. 
% Input PD image and a threshold between 0,1.
% output a 2D image mask. 
%===============================================
% (c) Mo Golbabaee m.golbabaee@bath.ac.uk, 2021
%===============================================

pd = abs(PD);
pd = pd/max(pd(:));
    
pd(pd<thresh)=0;
mask= pd;
mask= imfill(mask,8,'holes');
mask(mask>0) = 1;
