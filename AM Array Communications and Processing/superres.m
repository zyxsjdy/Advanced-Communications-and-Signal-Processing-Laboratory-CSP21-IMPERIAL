function [patternSuperres, weight] = superres(array, dirTarget, doa)
% Function: 
%   - obtain the directional gains of the superresolution beamformer
%
% InputArg(s):
%   - array: coordinates of the receiving sensors
%   - dirTarget: directions of the desired user
%   - doa: direction of arrival
%
% OutputArg(s):
%   - patternSuperres: directional gains in all directions
%   - weight: the weight on receiving antennas
%
% Comments:
%   - superresolution beamformers maximise the SIR, which is optimum to
%   suppress interference
%
% Author & Date: Yang (i@snowztail.com) - 27 Nov 18
mainlobe = [];
azimuth = 0: 180;
elevation = 0;
isLog = true;
dirInterf = setdiff(doa, dirTarget, 'rows');
spvInterf = spv(array, dirInterf, mainlobe);
spvTarget = spv(array, dirTarget, mainlobe);
spvAll = spv(array, [azimuth', elevation * ones(size(azimuth'))], mainlobe);
weight = fpoc(spvInterf) * spvTarget;
patternSuperres = abs(weight' * spvAll);
if isLog == true
    patternSuperres = 10 * log10(patternSuperres);
end
end

