function [output] = log_pdf(x,mu,covariance)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(x,2);
output = zeros(1,N);
for i=1:N
    output(1,i) = (-1/2*(x(:,i)-mu)'/covariance*(x(:,i)-mu))- (1/2*numel(x(:,i))*log(2*pi)+1/2*log(det(covariance)));
end
end

