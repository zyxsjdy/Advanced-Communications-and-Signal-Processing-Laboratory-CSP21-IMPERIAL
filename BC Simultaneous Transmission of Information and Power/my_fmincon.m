
Nc = 10;
N0 = 1;
h = [0.1+0.1i 0.2+0.8i 0.01+0.2i 0.1+0.9i 0.3+0.1i 0.1+0.7i 0.09+0.02i 0.1+0.8i 0.4+0.8i 0.1+0.3i];
Pn = zeros(1,Nc);
%P_lim = 10.3786; % for lambda = 0.28
P_lim = 1;

x0 = zeros(1,Nc);
A = [-1 0 0 0 0 0 0 0 0 0;
     0 -1 0 0 0 0 0 0 0 0;
     0 0 -1 0 0 0 0 0 0 0;
     0 0 0 -1 0 0 0 0 0 0;
     0 0 0 0 -1 0 0 0 0 0;
     0 0 0 0 0 -1 0 0 0 0;
     0 0 0 0 0 0 -1 0 0 0;
     0 0 0 0 0 0 0 -1 0 0;
     0 0 0 0 0 0 0 0 -1 0;
     0 0 0 0 0 0 0 0 0 -1;
     1 1 1 1 1 1 1 1 1 1];
b = [0;0;0;0;0;0;0;0;0;0;P_lim];

[x,~] = fmincon(@p,x0,A,b,[],[])
sum(x)

badness = N0./(abs(h)).^2;
b = bar([badness',x'],1,'stacked','FaceColor','flat');
for i = 1:Nc
    colour_b(i,:) = [0 0 1];
    colour_p(i,:) = [1 0 0];
end
b(1).CData = colour_b;
b(2).CData = colour_p;

legend('Noise to Carrier Ratio','Power allocation','FontSize',11); % Set the legend
xlabel('Subchannel Indices','FontSize',13); % Set the axes
title('Water-Filling Algorithm (λ = 0.28)','FontSize',15); % Set the title

text(7, 0.5, sprintf('TimZ\n15/11/2021'),'FontSize',12);























