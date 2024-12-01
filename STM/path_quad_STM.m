function [Obj, Acc, Data,time]= path_quad_STM(idx)
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
para.Size=[5,5,224];
para.Cp=10.^[-1:0.1:1];
para.tol_error=1e-4;
para.max_iter=10000;
for i=1:3
    para.w{i}=randn(1,para.Size(i));
    para.w{i}=para.w{i}/norm(para.w{i});
end
%% 开始循环
obj.w = para.w;
W = outprod(obj.w);
alpha_int=zeros(length(Y),1);time=zeros(1,length(para.Cp));
for  j=1:length(para.Cp)
    obj.iter = 1;
    obj.error(obj.iter) = 10;
    while obj.error(obj.iter) > para.tol_error && obj.iter < para.max_iter
        old_norm = frob(W); 
        for i = 1 : size(para.Size, 2)
            tem_data = zeros(length(Y), para.Size(i));
            k_mode_coef = obj.w;
            k_mode_coef(i) = [];
            j_iter = 1:size(para.Size, 2);
            j_iter(i) = [];
            for k = 1:length(Y)
                need_data = X{k};
                for l = j_iter
                    if l < i
                        need_data = tmprod(need_data,k_mode_coef{l},l);
                    else
                        need_data = tmprod(need_data,k_mode_coef{l-1},l);
                    end
                end
                tem_data(k,:) = need_data;
            end
            kmode_coef_tensor = outprod(k_mode_coef);
            norm_fix = frob(kmode_coef_tensor);
            tem_data = [tem_data ones(size(tem_data,1),1)];
            if j>1
                alpha_int=para.Cp(j)/para.Cp(j-1)*model.x;
            end
            model= DCDM1(tem_data,Y,norm_fix,para.Cp(j),alpha_int);
            time(j)=time(j)+model.t_dcdm;
            w_new = model.w';
            obj.w{i} = w_new(1:para.Size(i));
            obj.bias = w_new(para.Size(i)+1);
        end
        W = outprod(obj.w);
        obj.iter = obj.iter + 1;
        obj.error(obj.iter) = abs(old_norm - frob(W));
    end

    %测试
    test_data=zeros(length(Ytest),1);
    for l = 1:length(Ytest)
         sample = Xtest{l};
         for s = [1,2,3]
              sample = tmprod(sample,obj.w{s},s);
         end
         test_data(l,1) = sample+obj.bias;
    end
    Y_pre = sign(test_data);
    Y_pre(Y_pre==0) = 1;
    Acc(j) = mean(Y_pre==Ytest);
    Obj(j)=obj;
%     save('cutting_rank1STM1.mat','Acc','time')
end

