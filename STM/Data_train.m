function  [Data]=Data_train(X_pos,X_neg,idx,i)
idx_pos = idx.pos;
idx_neg = idx.neg;
Data.X_test = [X_pos(idx_pos==i,:); X_neg(idx_neg==i,:)];
Data.X_train = [X_pos(idx_pos~=i,:); X_neg(idx_neg~=i,:)];
Data.Y_train = [ones(sum(idx_pos~=i),1); -ones(sum(idx_neg~=i),1)];
Data.Y_test = [ones(sum(idx_pos==i),1); -ones(sum(idx_neg==i),1)];
end