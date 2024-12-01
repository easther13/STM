function [Obj, Acc]= path_quad_STM(X,Y,para,X_test, Y_test)
%% 开始循环
obj.iter = 1;
obj.error(obj.iter) = 10;
obj.w = para.w;
W = outprod(obj.w);
for  j=1:length(para.Cp)
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
                for j = j_iter
                    if j < i
                        need_data = tmprod(need_data,k_mode_coef{j},j);
                    else
                        need_data = tmprod(need_data,k_mode_coef{j-1},j);
                    end
                end
                tem_data(k,:) = need_data;
            end
            kmode_coef_tensor = outprod(k_mode_coef);
            norm_fix = frob(kmode_coef_tensor).^2;
            tem_data = [tem_data ones(size(tem_data,1),1)];
            K = tem_data*tem_data';
            H = 1/norm_fix*diag(Y)*K*diag(Y);
            f = -ones(length(Y),1);
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb = zeros(length(Y),1);
            ub = para.Cp*ones(length(Y),1);
            t2 = tic;
            obj.alpha{obj.iter,i} = quadprog(H,f,A,b,Aeq,beq,lb,ub);
            obj.time_quad(obj.iter,i) = toc(t2);
            w_new = 1/norm_fix*sum(obj.alpha{obj.iter,i}.*Y.*tem_data);
            obj.w{i} = w_new(1:para.Size(i));
            obj.bias = w_new(para.Size(i)+1);
        end
        W = outprod(obj.w);
        obj.iter = obj.iter + 1;
        obj.error(obj.iter) = abs(old_norm - frob(W));%误差
    end

    %测试
    test_data=zeros(length(Y_test),1);
    for l = 1:length(Y_test)
         sample = X_test{l};%第k个样本
         for s = [1,2,3]%每次迭代,j取j_iter中的一个值
              sample = tmprod(sample,obj.w{s},s);
         end
         test_data(l,1) = sample+obj.bias;
    end
    Y_pre = sign(test_data);
    Y_pre(Y_pre==0) = 1;
    Acc(j) = mean(Y_pre==Y_test);
    Obj(j)=obj;
end

