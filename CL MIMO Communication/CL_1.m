%% Initialization
clc;
clear;
close all;

%% Set basic parameters
nr = [5 6 7 8 9 10]; % number of receive antennas
nt = nr; % number of transmit antennas

p = 31; % take 31 samples
SNR_dB = (0:30/(p-1):30); % SNR_dB = 0, 0.3, ... , 29.7, 30
SNR = 10.^(SNR_dB/10); % calculate SNR, SNR_dB = 10log10(SNR), SNR_dB = 1, ... , 1000

C_H = zeros(10000,p); % initialize capacity C_H under certain nr with 10000 rows, each row is calculated from a random H, with length p
C_ergodic = zeros(length(nr),p); % initialize mean capacity C_ergodic, each row is the mean of C_H under certain nr, with length p

% Avoid the pseudorandomness
rand('twister',mod(floor(now*8640000),2^31-1))

%% Calculation
for j = 1:length(nr) % take different nr = [5 6 7 8 9 10]
    
    I = eye(nr(j)); % build nr*nr identical matrix
    H = zeros(nr(j)); % initialize channel H (nr*nr)

    for L = 1:10000 % use i.i.d. random H to calculate C_H
        for m = 1:nr(j)
            for n = 1:nr(j)
                % find elements of H, H is a complex Gaussian random matrix with i.i.d. entries
                H(m,n) = (randn+1i*randn)/sqrt(2); % divided by sqrt(2) to keep variance = 1
            end
        end
        H_h = H'; % H_h is the conjugate transpose of H

        for k = 1:p % calculate C_H with H, H_h, and p values of SNR from 1-1000 (0-30dB)
            C_H(L,k) = log2(det(I+SNR(k)/nt(j).*(H*H_h)));
        end
    end
    C_ergodic(j,:) = mean(C_H); % C_ergodic is the average of 10000 rows of C_H
end

C_ergodic_abs = abs(C_ergodic); % take absolute value to plot the figure (the imaginary part is about 1.0e-16 degree, ignorable)

%% Plot the figure
set(gcf,'Position',[10,10,700,500]); % Set the figure size
set(gca,'fontname','Times New Roman'); % Set the font type

a = plot(SNR_dB,C_ergodic_abs(1,:),'LineWidth',1.5); % Plot the figure
hold on
b = plot(SNR_dB,C_ergodic_abs(2,:),'LineWidth',1.5);
c = plot(SNR_dB,C_ergodic_abs(3,:),'LineWidth',1.5);
d = plot(SNR_dB,C_ergodic_abs(4,:),'LineWidth',1.5);
e = plot(SNR_dB,C_ergodic_abs(5,:),'LineWidth',1.5);
f = plot(SNR_dB,C_ergodic_abs(6,:),'LineWidth',1.5);

a.Color = [1 0 0]; % Set the colour of lines
b.Color = [0 0.9 1];
c.Color = [1 0 1];
d.Color = [1 0.7 0];
e.Color = [0 0.8 0];
f.Color = [0 0 0.8];

grid on
legend('nt = nr = 5','nt = nr = 6','nt = nr = 7','nt = nr = 8','nt = nr = 9','nt = nr = 10','FontSize',11); % Set the legend
xlabel('SNR (dB)','FontSize',13); % Set the axes
ylabel('Capacity (bps/Hz)','FontSize',13);
title('MIMO Capacity vs SNR, with different number of antennas','FontSize',15); % Set the title
text(22, 15, sprintf('Yuxiang Zheng\n17/10/2021'),'FontSize',12);

