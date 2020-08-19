%Computation of JACCARD INDEX
function [JaccardIdx]=JCC(x,xhat,k)
% x is the true vector
% xhat the estimated one
% k is the number of sources

x_bin = zeros(length(x),1);
[~,locs1] = findpeaks(abs(x),'NPeaks',k);
x_bin(locs1)= 2;
% non-null coefficients equals to 2

xhat_bin = zeros(length(xhat),1);
xhat = xhat ./ max(xhat);
[~,locs2] = findpeaks(abs(xhat),'MinPeakProminence',0.1);
xhat_bin(locs2)= -1;
% forced non-null coefficient to -1


% So : x + xhat = 1 if well detected
%                -1 if false detection
%                 2 if missed

MISSED = sum((x_bin+xhat_bin)==2);
FD     = sum((x_bin+xhat_bin)==-1);
GD     = sum((x_bin+xhat_bin)==1);

JaccardIdx = GD / (GD+FD+MISSED) ;
if isnan(JaccardIdx)
    JaccardIdx=0;
end
end