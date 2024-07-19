function [y,t,x,v]=mb1_testsig(xfa,vfa,snr,n,fs)
% MB1_TESTSIG generate a test signal for MB1 experiment
%
%  Inputs: xfa(m,2) each row gives the frequency and complex phasor amplitude
%                   of a tonal component of the signal.
%                   E.g. [100 -2i] gives a 100Hz sine wave of amplitude 2
%          vfa(k,2) as above but for tonal noise components
%          snr      snr of white noise component
%          n        number of samples 
%          fs       sample frequency [default = 1]
%
% Outputs: y(n,1)   noisy output signal
%          t(n,1)   sample times
%          x(n,1)   clean signal
%          v(n,1)   noise signal
if nargin<5
    fs=1;
end
[x,t]=mb1_tones(xfa,n,fs);
px=xfa(:,2)'*xfa(:,2)/2;        % mean power of x
if isempty(snr) || snr==Inf
    v=mb1_tones(vfa,n,fs);
else
    v=mb1_tones(vfa,n,fs)+10^(-snr/20)*sqrt(px)*randn(n,1);
end
y=x+v;
if ~nargout
    plot(t,[x,y]);
    axisenlarge([-1 -1.05]);
    legend('Clean','Noisy','location','northeast');
    xlabel(['Time (' xticksi 's)']);
end