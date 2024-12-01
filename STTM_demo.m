% STTM demo on cifar-10

% clear all

load HSI_Salines.mat

class_num=2;
% para.Cp=2.^[-3:1:3];
para.Cp=2.^(-3);
W_r=[ones(4,1),[2:5]',[2:5]',ones(4,1)]; 
%Salines
samplenum=30;%训练集个数
validnum=10;%验证集个数
label(label==0) = -1;
X_pos=X(label==1,:);
X_Neg=X(label==-1,:);
X=[X_pos;X_Neg];
positive_indices=randperm(50,40);
negative_indices=randperm(50,40)+50;
remaining_indices=setdiff(1:100,[positive_indices,negative_indices]);
trainX=X([positive_indices,negative_indices],:);
testX=X(remaining_indices,:);



for s=1:1
for j=1:length(para.Cp)
for loop=1:4
%% model training
% %Salines
traindata=[trainX(1:samplenum(s),:);trainX(41:samplenum(s)+40,:)];
trainingL=[ones(samplenum(s),1);-ones(samplenum(s),1)]; 
[e,labels,W,b]=f2_STTM(traindata,trainingL,W_r(loop,:),para.Cp(j));

%% validation
%Salines
validdata=[trainX(samplenum(s)+1:samplenum(s)+10,:);trainX(samplenum(s)+41:end,:)];
validL=[ones(validnum(s),1);-ones(validnum(s),1)]; 
X=validdata;
N=size(X,1);
X=cat(4, X{:});%将训练集重新叠加
X=reshape(permute(X, [4 2 3 1]),20,5,5,224);
X=permute(X,[2 3 4 1]);%7*4*4*7*60000 tensor   有没有必要要换？
X=tt_tensor(X,1e-2); %do not do truncation here first, but can use TT-feature to replace to save the storage and computation
valid_X=full(X,[5600 N]);
valid_X=valid_X';


W_new=full(cell2core(tt_tensor(1),W));
A=valid_X*W_new+b;
% A=valid_X*W_new;
B=zeros(size(valid_X,1),1);
B(A<0)=-1;B(A>0)=1;  %need to care here

diff=B-validL;
diff(diff~=0)=1;
error=sum(diff)/N;
valid_error(loop,s)=error;


end
% end

[~,ind]=min(valid_error);
% %Salines
traindata=[trainX(1:samplenum(s),:);trainX(41:samplenum(s)+40,:)];
trainingL=[ones(samplenum(s),1);-ones(samplenum(s),1)]; 
[e,labels,W,b]=f2_STTM(traindata,trainingL,W_r(loop,:),para.Cp(j));




%% test
% % Salines
X=testX;
testingL=[ones(10,1);-ones(10,1)]; 
N=size(X,1);
X=cat(4, X{:});%将训练集重新叠加
X=reshape(permute(X, [4 2 3 1]),20,5,5,224);%N*61*73*61 tensor
X=permute(X,[2 3 4 1]);%7*4*4*7*60000 tensor
X=tt_tensor(X,1e-2); %do not do truncation here first, but can use TT-feature to replace to save the storage and computation
test_X=full(X,[5600 N]);
test_X=test_X';


W_new=full(cell2core(tt_tensor(1),W));
A=test_X*W_new+b;
% A=test_X*W_new;
B=zeros(size(test_X,1),1);
B(A<0)=-1;B(A>0)=1;  %need to care here

diff=B-testingL;
diff(diff~=0)=1;
error=sum(diff)/N;
test_error(j,s)=error;
Acc(j)=1-error
end
end
