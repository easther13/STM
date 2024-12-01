function [Bestcv,Besttimetr,Besttimete] = KTTCPMain_lib(X,label,l,data_KTTCP,t)
% deatil of the inputs are as following:

%          X           the input data cell array, n * 1 --- each array represents a three-order tensor 
%                      n is the number of training examples
%          label       the output labels associated with the input data, n * 1
%          l           the rank of tensor decomposition  
%          data_KTTCP: kernel projected TT-CP expansion data 
%          t           number of repitition of whole procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          c1, c2 :  the trade-off parameter range [2^c1, 2^(c1+1),..., 2^(c2-1), 2^c2] in SVM model
%          g1, g2 :  the RBF kernel width parameter range [2^g1, 2^(g1+1),..., 2^(g2-1), 2^g2] in SVM model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
% Add Path
addpath('.\libsvm-master');
%% Initialize
n=size(X,1);                                                                    % Row is data number
c=ones(1,n);  
a=cumsum(c);

c1=-8;    
c2=8;
g1=-8;   
g2=8;
% c1=3;c2=3;
% g1=5;g2=5;

acc=0;
counttimetr=0;                                                                  % Training time
counttimete=0;                                                                  % Test time
rand('state',0);  % randomseed，确保在接下来的随机数生成中使用了相同的种子
%% Train and test
% Repeat t times with k-fold cross validation 
%t=1;
k=5;                                    
for i=1:t                                                                        % Repeat t times
    randseed = round(rand(1)*5489); 
    elimin_test=Divide(label,k,randseed);                                        
    b=setdiff(a,elimin_test{k,1});  
    Y=X(b,:);  %训练集
    Y_label=label(b);
    randseed = round(rand(1)*5489);  %生成一个新的随机数种子
    Div=Divide(Y_label,k,randseed);                                             
    [bestcv, bestc,bestg, ~,~]= TrainAvgAcuTTCP_lib(Y,Y_label,l,k,data_KTTCP(b,1),Div,c1,c2,g1,g2);  % Train and select the optimal paremeters 
    for j=1:5                                                                    % Extra repeat 5 times to get more stable result
        randseed = round(rand(1)*5489);
        DivOpti=Divide(label,k,randseed);% 20% of the whole data
        [cv, ~,~,timetr,timete]=TrainAvgAcuTTCP_lib(X,label,l,k,data_KTTCP,DivOpti,bestc,bestc,bestg,bestg);  
        acc=acc+cv;
        counttimetr=counttimetr+timetr;
        counttimete=counttimete+timete;
    end
    fprintf('The accuracy is %g corresponding to the %g th repeat, bestc is %g, bestg is %g\n',acc/(i*j),i, bestc,bestg);
    clear elimin_test b Y Y_label Div bestc bestg
end

Bestcv = acc/(i*j);
Besttimetr=counttimetr/(i*j);
Besttimete=counttimete/(i*j);
%Bestmodel = model;
end