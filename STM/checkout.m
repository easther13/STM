function [ alpha ] = checkout( H, alpha, C, FlagReduce)
%CHECKOUT 此处显示有关此函数的摘要
%   此处显示详细说明
%%HRSTM
    eps = 1e-2;
    m = length(FlagReduce);
    for k=1:500 
        alphak=alpha;
        for i = 1:m
            G = H(i,:)*alpha-1;
            alpha(FlagReduce(i)) = min(max(alpha(FlagReduce(i))-G/H(i,i),0),C);
        end
        k = k+1 ;
        Diff = alpha-alphak;
        if norm(Diff) < eps 
            break;
        end
    end
end


