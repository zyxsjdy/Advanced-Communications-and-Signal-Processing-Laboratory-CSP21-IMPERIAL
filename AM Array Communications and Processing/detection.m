function [nSources, eigVectorSignal] = detection(covRx)
% Function: 
%   - detector for continous signal, based on eigendecomposition
%
% InputArg(s):
%   - covRx: covariance matrix of the received signal
%
% OutputArg(s):
%   - nSources: number of sources
%   - eigVectorSignal: signal eigenvector
%
% Restraints:
%   - Traditional detection is based on observations, where the criterion
%   of regarding a eigenvalue as 'small' corresponding to noise is
%   determined by human. In this case we use a simplified model that based
%   on comparison: the threshold is defined by the minimum eigenvalue and a
%   ratio. This ratio should be determined carefully. The precision of this
%   function needs improvements.
% Comments:
%   - also obtain signal eigenvector to create subspace for MUSIC algorithm
%
% Author & Date: Yang (i@snowztail.com) - 27 Nov 18
[eigVector, eigValue] = eig(covRx);
eigValue = abs(diag(eigValue));
% assume max noise power / min noise power is below this ratio
% TO BE DESIGNED BY ACTUAL CASES
if min(eigValue) >= 1e-5
    noiseRatioThr = 1.5;
else
    noiseRatioThr = 1e4;
end
eigNoiseThr = min(eigValue) * noiseRatioThr;
nSources = sum(eigValue > eigNoiseThr);
eigVectorSignal = eigVector(:, eigValue > eigNoiseThr); 
% do not use noise eigenvector directly for noise subspace
% noiseEigVector = eigVector(:, eigValue <= eigNoiseThr);
end

