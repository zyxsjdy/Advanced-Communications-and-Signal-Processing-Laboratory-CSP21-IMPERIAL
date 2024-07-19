function [doa] = music(array, covRx)
% Function: 
%   - find the direction of arrival based on MUSIC algorithm
%
% InputArg(s):
%   - array: coordinates of the receiving sensors
%   - covRx: covariance matrix of the received signal
%
% OutputArg(s):
%   - doa: direction of arrival in degree
%
% Comments:
%   - noise eigenvectors are orthogonal to array manifold
%   - the value of the cost function should approach zero at doa
%
% Author & Date: Yang (i@snowztail.com) - 27 Nov 18
mainlobe = [];
azimuth = 0: 180;
elevation = 0;
costFun = zeros(length(azimuth), 1);
[nSourses, eigVectorSignal] = detection(covRx);
for iAzimuthAngle = azimuth
    spvComponent = spv(array, [iAzimuthAngle elevation], mainlobe);
    costFun(iAzimuthAngle + 1) = spvComponent' * fpoc(eigVectorSignal) * spvComponent;
end
[~, doaIndex] = mink(costFun, nSourses);
doaAzimuth = doaIndex - 1;
doa = [doaAzimuth elevation * ones(size(doaAzimuth))];
doa = sortrows(doa, 1);
end