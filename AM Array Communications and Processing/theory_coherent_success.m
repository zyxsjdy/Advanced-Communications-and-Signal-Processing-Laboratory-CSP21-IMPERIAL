clear; close;
array=[-2 0 0; -1 0 0; 0 0 0; 1 0 0; 2 0 0];
azimuth = 0: 180;
elevation = 0;
directions = [30 0; 35 0; 90 0];
dirTarget = directions(3, :);
gainWh = 1;
% additive uncorrelated isotropic noise, SNR = 40 dB
varNoise = 1e-4;
%% Array pattern plot
% only determined by array; best direction is 90 Azimuth
patternPlain = patten(array);
% max gain at target direction
plot2d3d(patternPlain, azimuth, elevation);
hold on;
%% Covariance matrix: theory
% normalised equal power sources -> unit diagonal; size determined by
% number of transmitters
% two sources are fully correlated
covCoherentTx = [1 1 0; 1 1 0; 0 0 1];
% averaged covariance matrix of transmitted signal by spatial smoothing
[covMeanRx, subarray] = spatial_smoothing(array, directions, covCoherentTx, varNoise);
%% source information is actually unknown
directions = [];
covCoherentTx = [];
varNoise = [];
%% Estimation: superresolution approach
% detect doa with any subarray
doa = music(subarray{1}, covMeanRx);
[patternSuperres, ~] = superres(array, dirTarget, doa);
plot2d3d(patternSuperres, azimuth, elevation, 'Gain in dB', 'Gain pattern by beamformers');
legend('Plain', 'Superresolution','location','southeast');
