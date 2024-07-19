%% Initialization
clc;
clear;
close all;

%% Set basic parameters
n = 2; % number of receive antennas
p = 8; % take 101 samples
Loop = 1000;
b = 10^6; % transmit 10^6 bits

I = eye(n,n);

SNR_dB = (0:35/(p-1):35); % SNR_dB = 0, 0.35, ... , 34.65, 35
SNR = 10.^(SNR_dB/10); % calculate SNR, SNR_dB = 10log10(SNR), SNR_dB = 1, ... , 3162.3

N_L = zeros(n,b/2);
ML_coded = zeros(16,0);

BER_ML_coded = zeros(Loop,p);
BER_ZF = zeros(Loop,p);
BER_MMSE = zeros(Loop,p);

mean_BER_ML_coded = zeros(1,p);
mean_BER_ZF = zeros(1,p);
mean_BER_MMSE = zeros(1,p);

%% Gray Mapping

x_00 = sqrt(0.5)*(-1-1i); % four types of symbols
x_01 = sqrt(0.5)*(-1+1i);
x_10 = sqrt(0.5)*(+1-1i);
x_11 = sqrt(0.5)*(+1+1i);

x_gray = [[x_00 x_00 x_00 x_00 x_01 x_01 x_01 x_01 x_10 x_10 x_10 x_10 x_11 x_11 x_11 x_11];
          [x_00 x_01 x_10 x_11 x_00 x_01 x_10 x_11 x_00 x_01 x_10 x_11 x_00 x_01 x_10 x_11]];
x_gray_coded = zeros(n,n,16);
for q = 1:16
    x_gray_coded(:,1,q) = x_gray(:,q);
    x_gray_coded(:,2,q) = [-conj(x_gray(2,q));conj(x_gray(1,q))];
end

c_all =  sqrt(0.5)*(randsrc(n,b/4)+1i*randsrc(n,b/4));

%% Plot the figure
set(gcf,'Position',[10,10,700,500]); % Set the figure size
set(gca,'fontname','Times New Roman'); % Set the font type

f_1 = semilogy(SNR_dB,mean_BER_ML_coded,'LineWidth',1.5); % Plot the figure
axis([min(SNR_dB) max(SNR_dB) 1/b 1])

f_1.Color = [1 0 0]; % Set the colour of lines

grid on
legend('BER ML','FontSize',11); % Set the legend
xlabel('SNR (dB)','FontSize',13); % Set the axes
ylabel('BER','FontSize',13);
title('BER of ML versus SNR','FontSize',15); % Set the title
text(6, 0.3*10^-4, sprintf('TimZ\n17/10/2021'),'FontSize',12);
drawnow

f_1.YDataSource = 'mean_BER_ML';

%% Main Loop
for L = 1:Loop

    H = sqrt(0.5)*(randn(n,n)+1i*randn(n,n));
    N_L = randn(n,b/2)+1i*randn(n,b/2);

    for k = 1:p
        E_ML_coded = 0;
        E_ZF = 0;
        E_MMSE = 0;

        %noisePower = 1/(10^(SNR(k)/10));
        N_p = sqrt(0.5)*N_L;
        %N_p = sqrt(noisePower/2)*N_L;

        for o = 1:b/4
            c = c_all(:,o);
            c_coded(:,1) = c;
            c_coded(:,2) = [-conj(c(2));conj(c(1))];

            N_coded = N_p(:,(2*o-1):2*o);
            A = sqrt(SNR(k)/n);
            y_coded = A*H*c_coded + N_coded;

            for q = 1:16
                ML_coded(q) = (norm(y_coded-A*H*x_gray_coded(:,:,q), 'fro'))^2;
            end
            [~,Col]=min(ML_coded);
            R_ML_coded = x_gray(:,Col);

            if (sign(real(R_ML_coded(1,1))) ~= sign(real(c(1,1)))) || (sign(imag(R_ML_coded(1,1))) ~= sign(imag(c(1,1))))
                E_ML_coded = E_ML_coded + 1;
            end
            if (sign(real(R_ML_coded(2,1))) ~= sign(real(c(2,1)))) || (sign(imag(R_ML_coded(2,1))) ~= sign(imag(c(2,1))))
                E_ML_coded = E_ML_coded + 1;
            end
        end

        BER_ML_coded(L,k) = E_ML_coded/b;
    end

    mean_BER_ML_coded = mean(BER_ML_coded(1:L,:),1);
    refreshdata
    drawnow
end


























