function [nSources,aicFun] = detector_aic(nSnapshots, covSampleRx)
% Function: 
%   - detector for practical sampled signal, based on akaike information
% criterion
%
% InputArg(s):
%   - nSnapshots: number of samples
%   - covSampleRx: covariance matrix of the practical signal
%
% OutputArg(s):
%   - nSources: number of sources
%
% Comments:
%   - source count equals minimum index of the function minus one
%
% Author & Date: Yang (i@snowztail.com) - 27 Nov 18
[~, eigValue] = eig(covSampleRx);
eigValue = sort(abs(diag(eigValue)));
nReceivers = length(eigValue);
aicFun = (-2) * nSnapshots * (log(flip(cumprod(eigValue))) + (nReceivers: -1: 1)' .* (log((nReceivers: -1: 1)') - log(flip(cumsum(eigValue))))) ...
    + 2 * (0: nReceivers - 1)' .* (2 * nReceivers: -1: nReceivers + 1)';
[~, minPos] = min(aicFun);
nSources = minPos - 1;
end

