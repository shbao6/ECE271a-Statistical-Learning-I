function [mean] = guassian_mle(x)
%x is the data
[m,p] = size(x);
mean = zeros(1,p);
for i = 1:m
    mean = mean + x(i,:);
end 
mean = mean/m;


