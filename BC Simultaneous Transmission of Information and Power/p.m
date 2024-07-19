function [outputArg1] = p(x)

lambda = 0.28;
Nc = 10;
N0 = 1;
h = [0.1+0.1i 0.2+0.8i 0.01+0.2i 0.1+0.9i 0.3+0.1i 0.1+0.7i 0.09+0.02i 0.1+0.8i 0.4+0.8i 0.1+0.3i];
y = 0;

for i = 1:Nc
    y = y - log(1+x(i)*(abs(h(i)))^2/N0);

end

outputArg1 = y;
end

