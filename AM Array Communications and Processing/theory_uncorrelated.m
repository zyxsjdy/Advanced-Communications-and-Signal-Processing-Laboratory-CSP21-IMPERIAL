clear; close;
array=[-2 0 0; -1 0 0; 0 0 0; 1 0 0; 2 0 0];
azimuth = 0: 180;
elevation = 0;
directions = [30 0; 35 0; 90 0];
% assume target is the last direction
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
%% Source position vector
% gain of elements on all directions
spvSources = spv(array, directions);
%% Covariance matrix: theory
% normalised equal power sources -> unit diagonal; uncorrelated -> others
% entries zero; size determined by number of transmitters
covTx = eye(length(directions));
% covariance matrix of transmitted signal
% diagonal -> signal power; others -> covariance of signals
covRx = spvSources * covTx * spvSources' + varNoise * eye(length(array));
%% source information is actually unknown
directions = [];
covTx = [];
spvSources = [];
varNoise = [];
%% Estimation: conventional approach
% manifold vector of the desired source
spvTarget = spv(array, dirTarget);
% optimum weight by Wiener-Hopf solution
weightWh = gainWh * covRx \ spvTarget;
patternWh = patten(array, weightWh);
plot2d3d(patternWh, azimuth, elevation);
hold on;
%% Estimation: superresolution approach
% estimate doa with MUSIC algorithm
doa = music(array, covRx);
[patternSuperres, ~] = superres(array, dirTarget, doa);
plot2d3d(patternSuperres, azimuth, elevation, 'Gain in dB', 'Gain pattern of beamformers');
legend('Plain', 'Wiener-Hopf', 'Superresolution','location','southeast');
