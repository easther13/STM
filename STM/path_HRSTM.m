function [Obj, Acc,time,T2]= path_HRSTM(idx)
load('HSI_Salines.mat');
Y=label;
Y(Y==0) = -1;
X_pos=X(Y==1,:);
X_Neg=X(Y==-1,:);
testPos=[X_pos(idx.pos==1,:)];trainPos=[X_pos(idx.pos~=1,:)];
testNeg=[X_Neg(idx.neg==1,:)];trainNeg=[X_Neg(idx.neg~=1,:)];
Data.X_test = [testPos; testNeg];
Data.X_train = [trainPos; trainNeg];
Data.Y_train = [ones(length(trainPos),1);-ones(length(trainNeg),1)];
Data.Y_test = [ones(length(testPos),1);-ones(length(testNeg),1)];
X=Data.X_train;Y=Data.Y_train; Xtest =Data.X_test;Ytest = Data.Y_test;
m=size(X,1);
para.Size=[5,5,224];
% para.Cp=2.^[-1:0.1:1];
para.Cp=2.^(-0.7);
para.tol_error=1e-3;
para.max_iter=10000;
para.coresize=[5,5,100];
for i=1:3
%     para.W{i}=randn(para.Size(i),floor(para.Size(i)/2));%factor matrix
    para.W{i}=randn(para.Size(i),para.coresize(i));%factor matrix
    para.W{i}=para.W{i}/norm(para.W{i},'fro');
end
% S{1}=kron(para.W{3},para.W{2});
% S{2}=kron(para.W{3},para.W{1});
% S{3}=kron(para.W{2},para.W{1});
% para.coresize=[size(para.W{1},2),size(para.W{2},2),size(para.W{3},2)];
para.G=randn(prod(para.coresize),1);
para.G=reshape(para.G,para.coresize);%G是单位张量
G_norm = frob(para.G);
para.G=(1/G_norm).*para.G;
%% 开始循环
obj.iter = 1;time=zeros(1,length(para.Cp));
obj.error(obj.iter) = 10;%初始误差值
W=tmprod(para.G,para.W,[1,2,3]);
alpha_int = zeros(m,1);Alpha_int = zeros(m,1);
for  j=1:length(para.Cp)%超参选择
    obj.iter=1;obj.error(obj.iter)=10;
    T1=tic;
    while obj.error(obj.iter) > para.tol_error && obj.iter < para.max_iter
        old_norm = frob(W); 
        %SOLVE FACTORE MATRIX
        for i = 1 : size(para.Size, 2)
            S=para.W;
            G{i}=tenmat(para.G,i).data;
            S(i)=[];S_i=kron(flip(S));
            P{i}=G{i}*S_i';
            for k = 1:length(Y)
                need_data = tenmat(X{k},i).data;
%                 need_data= need_data*P{i}';
%                 need_data = [need_data,ones(size(need_data,1),1)];
                need_datas{k,1} = need_data';
            end
            D=P{i};
            fun_D=@(x)D;
            cell_D=cell(m,1);%m是样本个数
            cell_D=cellfun(fun_D,cell_D,'UniformOutput',false);
            Q=cellfun(@mtimes, cell_D,need_datas, 'UniformOutput',false);
            Q=cellfun(@(x)[x;ones(1,size(x,2))],Q,'UniformOutput',false);
            P{i}=[P{i};ones(1,size(P{i},2))];
            K=P{i}*P{i}';
            K=sign(K) .* abs(K) .^ (-1/2);
            fun_K=@(x)K;
            cell_D=cell(m,1);
            cell_D=cellfun(fun_K,cell_D,'UniformOutput',false);
            X_A=cellfun(@mtimes,cell_D,Q,'UniformOutput',false);
            X_A=cell2mat(cellfun(@(x) reshape(x', [], 1), X_A', 'UniformOutput', false));
            H= diag(Y)*(X_A'*X_A)*diag(Y);            
            [obj.alpha{obj.iter,i},model] = DCDMG1(H,m,para.Cp(j),alpha_int);
            time(j)=time(j)+model.time;
            D=K^(2);
            fun_D=@(x)D;
            cell_D=cell(m,1);%m是样本个数
            cell_D=cellfun(fun_D,cell_D,'UniformOutput',false);
            a=obj.alpha{obj.iter,i}.*Y;
            Q_new=cellfun(@transpose, Q, 'UniformOutput',false);
            Q_new=arrayfun(@(i)a(i)*Q_new{i},1:m,'un',0)';
            cellArray=cellfun(@mtimes,Q_new,cell_D,'UniformOutput',false);
            para.W{i}= sum(cat(3, cellArray{:}),3);
            para.W{i}=para.W{i}(:,1:end-1);
            obj.W{i} = para.W{i};
        end

       
        %SOLVE CORE TENSOR
        obj.allW=kron(flip(obj.W));
        for k = 1:length(Y)
            need_data1 = tenmat(X{k},1).data;
            need_data1 =need_data1(:);
            tem_data1{k,1} = need_data1;%列向量
        end
        D=obj.allW';
        fun_D=@(x)D;
        cell_D=cell(m,1);%m是样本个数
        cell_D=cellfun(fun_D,cell_D,'UniformOutput',false);
        T=cellfun(@mtimes,cell_D,tem_data1,'UniformOutput',false);
        T=cellfun(@(x)[x;ones(1, size(x, 2))], T, 'UniformOutput', false);%T{i}指的是所有的样本对应的T,T{i}是一个cell,其中每一个列向量
        obj.allW=[obj.allW,ones(size(obj.allW,1),1)];
        K_G=obj.allW'*obj.allW;
%         D_G=K_G^(-1/2);
        D_G=sign(K_G) .* abs(K_G) .^ (-1/2);
        fun_DG=@(x)D_G;
        cell_DG=cell(m,1);%m是样本个数
        cell_DG=cellfun(fun_DG,cell_DG,'UniformOutput',false);
        X_G=cellfun(@mtimes,cell_DG,T,'UniformOutput',false);%是个包含很多样本的cell,其中每一个是列向量
        X_G=cell2mat(cellfun(@(x) reshape(x, [], 1), X_G, 'UniformOutput', false));%转成列向量，不用单独算迹

        H= diag(Y)*(X_G'*X_G)*diag(Y);   
        [obj.Alpha{obj.iter},Model] = DCDMG1(H,m,para.Cp(j),Alpha_int);
        time(j)=time(j)+model.time;
        D_G=D_G^(2);
        fun_DG=@(x)D_G;
        cell_DG=cell(m,1);%m是样本个数
        cell_DG=cellfun(fun_DG,cell_DG,'UniformOutput',false);
        a=obj.Alpha{obj.iter}.*Y;
        tem_dataset=arrayfun(@(i)a(i)*T{i},1:m,'un',0)';
        CellArray=cellfun(@mtimes,cell_DG,tem_dataset,'UniformOutput',false);
        vector_G= sum(cat(3, CellArray{:}),3);%向量
        %若para.G是向量，则
        obj.bias=vector_G(end,1);
        obj.G = vector_G(1:end-1,:);
        obj.Gnew=reshape(obj.G, [para.coresize(1),para.coresize(2),para.coresize(3)]);
        para.G=obj.Gnew;
        W=tmprod(obj.Gnew,obj.W,[1,2,3]);
        obj.iter = obj.iter + 1;
        obj.error(obj.iter) = abs(old_norm - frob(W));%误差

    end
    T2(j)=toc(T1);

    %测试
%     test_data=zeros(length(Ytest),1);
    for l = 1:length(Ytest)
        test_data(l,1)=sum(W.*Xtest{l},'all')+obj.bias;
    end
    Y_pre = sign(test_data);
    Y_pre(Y_pre==0) = -1;
    Acc(j) = mean(Y_pre==Ytest);
    Obj(j)=obj;
end

end