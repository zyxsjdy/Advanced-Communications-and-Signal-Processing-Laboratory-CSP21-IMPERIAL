function mb1_plotpsd(v,fp,fs)
%PLOTPSD plot the power spectrum density
%
%  Inputs: v    vector containing time-domain signal
%          fp   plotting bandwidth or range [default fs/2]
%          fs   sample frequency [default = 1]
nt=length(v); % number of points
if nargin<3 || isempty(fs)
    fs=1;               % default sample frequency = 1
end
nf=200;                 % number of points to plot
if nargin<2 || isempty(fp)
    fp=fs/2;
end
if length(fp)==1
    fp=linspace(0,fp,nf);           % define the frequency axis
else 
    fp=linspace(fp(1),fp(2),nf);    % define the frequency axis
end
nw=min(nt,round(nf*fs/(fp(end)-fp(1))));  	% choose window length
win=kaiser(nw,9.5);                 % Use a Kaiser analysis window
win=win/sqrt(fs*sum(win.^2));       % scale to give psd in energy/Hz
z=enframe(v,win,round(nf/2))';      % Split signal into overlapping frames
[y,f]=zoomfft(z,round(fs/(fp(2)-fp(1))),nf); % Calculate the DFT
x=10*log10(mean(abs(y.^2),2));      % Convert to dB
plot(f*fs,x);                       % plot the power spectrum
axisenlarge([-1 -1.05]);            % make it fit the axis nicely
ylim=get(gca,'ylim');
set(gca,'ylim',[max(ylim(2)-60,ylim(1)), ylim(2)]); % limit to 60 dB range
xlabel('Frequency (Hz)');
ylabel('PSD (dB)');
