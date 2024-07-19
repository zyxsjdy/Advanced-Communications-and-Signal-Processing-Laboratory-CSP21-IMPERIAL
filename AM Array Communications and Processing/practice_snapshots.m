clear; close;
array=[-2 0 0; -1 0 0; 0 0 0; 1 0 0; 2 0 0];
azimuth = 0: 180;
elevation = 0;
directions = [30 0; 35 0; 90 0];
nSnapshots = 250;
varNoise = 1e-4;
%% Detection
% signal construction
signalSample = zeros(length(array), length(array) * nSnapshots);
spvSources = spv(array, directions);
covTx = eye(length(directions));
covRx = spvSources * covTx * spvSources' + varNoise * eye(length(array));
[eigVectorIdeal, eigValueIdeal] = eig(covRx);
snapshots = randn(nSnapshots, 1) + 1i * randn(nSnapshots, 1);
% sampling
for iSnapshot = 1: nSnapshots
    signalSample(:, (iSnapshot - 1) * length(array) + 1 : iSnapshot  * length(array)) = eigVectorIdeal * eigValueIdeal ^ (1/2) * snapshots(iSnapshot);
end
covSampleRx = 1 / nSnapshots * (signalSample * signalSample');
eig(covSampleRx)

[AIC_sourceNo] = detector_aic(nSnapshots, covSampleRx)
[MDL_sourceNo] = detector_mdl(nSnapshots, covSampleRx)









