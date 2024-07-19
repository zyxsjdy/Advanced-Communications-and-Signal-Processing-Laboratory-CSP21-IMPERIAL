

lambda = 0.2;
%mu = 0.2;
mu = 0.234;
%mu = 0.2439;
Nc = 10;
N0 = 1;
h = [0.1+0.1i 0.2+0.8i 0.01+0.2i 0.1+0.9i 0.3+0.1i 0.1+0.7i 0.09+0.02i 0.1+0.8i 0.4+0.8i 0.1+0.3i];
Pn = zeros(Nc,1);
badness = zeros(Nc,1);

for i = 1:Nc
    badness(i) = N0./(abs(h(i))).^2;
    Pn(i) = 1/(lambda-mu*(abs(h(i)))^2) - badness(i);
    if Pn(i)<0
        Pn(i) = 0;
    end
end

Pn'

b = bar([badness,Pn],1,'stacked','FaceColor','flat');
for i = 1:Nc
    colour_b(i,:) = [0 0 1];
    colour_p(i,:) = [1 0 0];
end
b(1).CData = colour_b;
b(2).CData = colour_p;

legend('Noise to Carrier Ratio','Power allocation','FontSize',11); % Set the legend
xlabel('Subchannel Indices','FontSize',13); % Set the axes
title('Optimal Power Allocations (λ = 0.2, μ = 0.244)','FontSize',15); % Set the title

text(1, 80, sprintf('TimZ\n15/11/2021'),'FontSize',12);



