function [idx]=cross_validation(X_pos,X_neg,K)
N_pos = size(X_pos,1);
N_neg = size(X_neg,1);
idx.pos = crossvalind('Kfold',N_pos,K);
idx.neg = crossvalind('Kfold',N_neg,K);
end