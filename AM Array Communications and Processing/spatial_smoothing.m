function [covMeanRx, subarray] = spatial_smoothing(array, directions, covCoherentTx, varNoise)
% Function: 
%   - find the mean covariance matrix by averaging subarray covariance
%   matrices to combat coherence
%
% InputArg(s):
%   - array: coordinates of the receiving sensors
%   - directions: directions of transmitted signals
%   - covCoherentTx: covariance matrix of the transmitted coherent signal
%   - varNoise: noise variance
%
% OutputArg(s):
%   - covMeanRx: mean covariance matrix
%   - subarray: coordinates of the subarray sensors
%
% Comments:
%   - the mean covariance matrix is with lower dimension
%   - pattern can be obtained by combining it with any valid subarray
%
% Author & Date: Yang (i@snowztail.com) - 27 Nov 18
% number of subarrays equals number of coherent signals
nSubarrays = length(covCoherentTx) - rank(covCoherentTx) + 1;
sizeSubarray = length(array) - nSubarrays + 1;
subarray = cell(nSubarrays, 1);
spvSubarray = cell(nSubarrays, 1);
covSum = 0;
for iSubarray = 1: nSubarrays
    subarray{iSubarray} = array(iSubarray: (iSubarray + sizeSubarray - 1), :);
    spvSubarray{iSubarray} = spv(subarray{iSubarray}, directions);
    covSum = covSum + spvSubarray{iSubarray} * covCoherentTx * spvSubarray{iSubarray}';
end
covMeanRx = covSum / nSubarrays + varNoise * eye(sizeSubarray);
