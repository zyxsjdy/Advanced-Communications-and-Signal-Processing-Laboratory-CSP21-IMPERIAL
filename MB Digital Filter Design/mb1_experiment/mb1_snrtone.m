function [r,a,e,z,x]=snrtone(y,xfa,fs)
%SNRTONE find SNR of one or more tones in noise
%  Inputs: y   noisy signal
%          xfa each row gives the frequency and complex phasor amplitude
%              of a tonal component of the signal.
%              E.g. [100 -2i] gives a 100Hz sine wave of amplitude 2
%          fs  sample frequency [default = 1]
%
% Outputs: r   estimated SNR (dB)
%          a   estimated phasor amplitudes
%          e   each row has the error in dB and phase for a tonal component
%          z   residual noise
%          x   estimated signal components
if nargin<3
    fs=1;
end
nfx=size(xfa,1);            % number of tones
nt=length(y);               % number of samples
t=(1:nt)'/fs;
a=zeros(nfx,1);             % space for phasor amplitudes
e=zeros(nfx,2);
for i=1:nfx
    fxi=xfa(i,1);           % tone frequency
    nti=min(nt,round(floor(nt*fxi/fs)*fs/fxi)); % number of samples in an exact number of cycles
    a(i)=exp(2i*pi*t(1:nti)*fxi)'*y(1:nti)*2/nti; % estimate phasor amplitude
    e(i,:)=[20*log10(abs(a(i)/xfa(i,2))) angle(a(i)/xfa(i,2))];
end
x=real(exp(2i*pi*t*xfa(:,1).')*a);  % estimated signal
z=y-x;                              % estimated noise
r=10*log10(a'*a/2/mean(z.^2));      % estimated SNR