function [Bestcv, Bestc,Bestg,time_tr,time_te]= TrainAvgAcuTTCP_lib(X,label,l,k,data_KTTCP,choose,c1,c2,g1,g2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Input
%          X          :  the input data cell array, n * 1 --- each array represents a three-order tensor
%                        n is the number of training examples
%          label      :  the output labels associated with the input data, n * 1
%          R          :  the rank of tensor decomposition
%          k          :  k-fold cross validation
%          data_KTTCP :  TTCP or KTTCP decomposition result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          c1, c2     :  the trade-off parameter range [2^c1, 2^(c1+1),..., 2^(c2-1), 2^c2] in SVM model
%          g1, g2     :  the RBF kernel width parameter range [2^g1, 2^(g1+1),..., 2^(g2-1), 2^g2] in SVM model

%%   Output:

%         Bestcv,Bestc,Bestg   :  Test accuracy obtained using k-fold cross validation in the optimal hyper-parameters (Bestc,Bestg)
%         time_tr              :  Training time     
%         time_te              :  Test time 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize
n=size(data_KTTCP,1);
global order
Order = 3;
c=ones(1,n);  
a=cumsum(c);
Bestcv=0;
Bestc=-100;
Bestg=-100;
Acctemp=zeros(k,c2-c1+1);
timetemp_tr=zeros(k,c2-c1+1);
timetemp_te=zeros(k,c2-c1+1);
time_tr=0;
time_te=0;

global l

%% k-fold cross validation
for log2g = g1:g2
    for cv=1:k
        b=setdiff(a,choose{cv,1}); % 80% of the whole data
        Ktrain=zeros(length(b),length(b)); %为什么又要分新的测试集和训练集，直接test不行吗？输出看一下
        Ktest=zeros(n-length(b),length(b));
        alpha_int=zeros(length(b),1);
        tic;
        for p=1:length(b)
            for q=1:p
                if Order~=1   
                   Ktrain(p,q)=Ker_fTTCP(data_KTTCP{b(p),1},data_KTTCP{b(q),1},Order,2^log2g,l);
                else
                   Ktrain(p,q)=Ker_fTTCP(X(b(p),:),X(b(q),:),Order,2^log2g);
                end
                if p~=q
                    Ktrain(q,p)=Ktrain(p,q);
                end
            end
        end
        
        time_tr1=toc;
        tic;
        for r=1:n-length(b)%Ktest不是方阵
            for p=1:length(b)
                if Order~=1
                   Ktest(r,p)= Ker_fTTCP(data_KTTCP{choose{cv,1}(1,r),1},data_KTTCP{b(p),1},Order,2^log2g,l);
                else
                   Ktest(r,p)= ker_fTTCP(X(choose{cv,1}(1,r),:),X(b(p),:),Order,2^log2g);
                end
            end
        end
        time_te1=toc;
        Ytrain=label(b);
        Htrain=diag(Ytrain)*Ktrain*diag(Ytrain);
        tempc=0;
        
         for log2c=c1:c2  
            tempc=tempc+1;
%             cmd=['-c ', num2str(2^log2c), ' -t ', num2str(4),' -q'];%设置了支持向量机训练的参数
            tic;
%             model{cv}= svmtrain(label(b), Ktrain1, cmd); % training the model，这个函数什么意思
            if log2c>c1
                alpha_int =2^log2c/2^(log2c-1)*model.x;
            end
            f = -ones(length(b),1);
            [alpha, model] = DCDMG(Htrain,length(b),2^log2c,alpha_int,Ktrain);
            time_tr2=toc;
            timetemp_tr(cv,tempc)=time_tr1+time_tr2;%使用 svmtrain 函数训练支持向量机模型
            tic;
%             [~,temp,~] = svmpredict(label(choose{cv,1}), Ktest1, model{cv},'-q'); % predicting the model
            Y=Ktest*(alpha.*label(b));
            temp=sign(Y);
            Acctemp(cv,tempc)=mean(temp==label(choose{cv,1}));
            time_te2=toc;
            timetemp_te(cv,tempc)=time_te1+time_te2;
%             Acctemp(cv,tempc)=temp(1)/100;  
        end % end for c
        clear b Ktrain Ktest %用于清除或清理 MATLAB 工作空间中的变量，以释放内存
    end %end for cv 
    Accvector=sum(Acctemp,1);%对 Acctemp 中的每列求和得到的
    timetrain=sum(timetemp_tr,1);
    timetest=sum(timetemp_te,1);
    [Acc,C]=max(Accvector);%C 是 Accvector 中最大值的索引，即最佳准确度对应的参数
    if Acc/k>Bestcv
        Bestcv=Acc/k;       
        Bestc=(C-(1-c1));
        Bestg= log2g;
        time_tr=timetrain(C)/k;
        time_te=timetest(C)/k;
    end
    fprintf('%g %g (best c=%g, g=%g, bestacc=%g) traintime=%g,testtime=%g\n',log2c, log2g, Bestc, Bestg, Bestcv,time_tr,time_te); %results
end % end for g
end

