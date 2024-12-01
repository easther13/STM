function [alpha, model] = DCDMG(Y,C,alpha_int, K)
 %% 初始化 
    times = 0;
%     maxIter = 20000;
%     maxDSRIter = 100;
    upperBound = C;
    lowerBound = 0;
    tstart=tic;
    Q=diag(K);
    QQ = Q;
    H=diag(Y)*K*diag(Y);
%     alpha = zeros(numSampleX, 1); %初始化alpha
    alpha=alpha_int;
%     w = zeros(size(X,2),1); %初始化w，w = X'*diag(Y)*alpha;
    %计算gap
    gap = alpha'*H*alpha + C*sum(max(0,1-H*alpha)) - sum(alpha);
%     gap = w'*w + C*sum(max(0,1-diag(Y)*X*w))  - sum(alpha);
    while times < 10000 && gap>1e-10
        %% 求解子问题
        alphak=alpha;
        for k = 1 : length(Y)   
            kk=k;
%             kk = L1(k);
%             S1 = gaussianKernel(X, X(kk,:), p);
%             Gk = Y(kk,:)*(alpha(L1,:).*Y(L1,:))'*S1-1;%同G
            Gk = H(kk,:)*alpha-1;
            % 使得Alpha一直在可行域    
            alpha(kk) = max([min([alpha(kk) - Gk./QQ(kk), upperBound]), lowerBound]);%这里对不对要试试
        end
%         w =w + X(L1)'*diag(Y(L1))*(alpha1-alphak(L1));
        times = times + 1;  %外循环次数更新
        Diff = alpha-alphak;
        if mod(times,10 )==0 
%         if mod(times,30 )==0 
%             tag = diag(Y)*X*w;
            gap = alpha'*H*alpha + C*sum(max(0,1-H*alpha)) - sum(alpha); %更新gap
%             gap = w'*w + C*sum(max(0,1-tag)) - sum(alpha);
        end  
        if times < 10000 && norm(Diff) < 1e-6 
            break;
        end
    end
    T = toc(tstart);model.x = alpha;model.time=T;model.realnumber=sum(alpha==0)+sum(alpha==C);
    times
end
