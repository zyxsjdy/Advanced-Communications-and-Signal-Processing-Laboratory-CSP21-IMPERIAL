function [x,t]=mb1_tones(fa,n,fs)
% MB1_TONES Generate a combination of sine wave tones
%
% Usage: fs=1000;                  % set sampling frequency = 1 kHz
%        x=mb1_tones([20 1; 30 -2i],2/fs,fs); % cosine wave @ 20 Hz + sine wave @ 30 Hz
%
%  Inputs: fa(m,2)  each row gives the frequency and complex phasor amplitude
%                   of a tonal component.
%                   E.g. [100 -2i] gives a 100Hz sine wave of amplitude 2
%          n        number of samples to generate
%          fs       sample frequency [default: 1]
%
% Outputs: x(n,1)   output waveform
%          t(n,1)   time of each sample starting at 0
%
if nargin<3 || isempty(fs)
    fs=1;
end
n=round(n); % ensure n is an integer
t=(0:n-1)'/fs;    % signal time axis
if isempty(fa)
    x=zeros(n,1);
else
    x=real(exp(2i*pi*t*fa(:,1)')*fa(:,2));
end
if ~nargout
    plot(t,x);
    xlabel(['Time (' xticksi 's)']);
end