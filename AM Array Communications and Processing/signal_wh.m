clear; close;
array=[-2 0 0; -1 0 0; 0 0 0; 1 0 0; 2 0 0];
azimuth = 0: 180;
elevation = 0;
gainWh = 1;
%% Signal recovery by Wiener-Hopf: with strong interference
load('xAudio'); load('xImage');
signalAudio = X_au;
signalImage = X_im;
soundsc(real(signalAudio(2, :)), 11025);
displayimage(signalImage(2, :), image_size, 201, 'The received signal at the 2nd antenna');
covAudio = signalAudio * signalAudio' / length(signalAudio(1, :));
covImage = signalImage * signalImage' / length(signalImage(1, :));
% doa are the same for audio and image;
doa = music(array, covAudio);
nTargets = length(doa);
for iTarget = 1: nTargets
    dirTarget = doa(iTarget, :);
    spvTarget = spv(array, dirTarget);
    weightWhAudio = gainWh * covAudio \ spvTarget;
    weightWhImage = gainWh * covImage \ spvTarget;
    recAudio = weightWhAudio' * signalAudio;
    recImage = weightWhImage' * signalImage;
    soundsc(real(recAudio), 11025);
    displayimage(recImage / max(recImage) * 256, image_size, 202, 'The received signal at o/p of W-H beamformer');
end
