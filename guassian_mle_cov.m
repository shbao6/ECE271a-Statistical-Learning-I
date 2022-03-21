function [covariance] = guassian_mle_cov(x)
%x is the data
[m,p] = size(x);
mean = zeros(1,p);
covariance = zeros(p,p);

for i = 1:m
    mean = mean + x(i,:);
end 
mean = mean/m;
for j = 1:m
    covariance = covariance + (x(j,:)-mean)' * (x(j,:)-mean);
end
covariance = covariance/m;

end

