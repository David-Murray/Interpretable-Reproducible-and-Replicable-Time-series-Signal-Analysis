function [ L,W] = set_L( feature, FD, c_k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

W = zeros(size(feature,1));

for j = 1:size(feature,2)
    
    W = W + c_k(j) * FD{j};
    
end
% W=W/99;
W = exp(-W);

W(W == diag(W)) = 0;

%  W = W - min(W(:));
%  W = W/max(W(:));

L = diag(sum(W))-W;

end



