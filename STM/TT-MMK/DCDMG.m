function [alpha, model] = DCDMG(H,m,C,alpha_int,K)
 %% 初始化 
    times = 0;
    upperBound = C;
    lowerBound = 0;
    tstart=tic;
    Q=diag(K);
    QQ = Q;
    alpha=alpha_int;
    %计算gap
    gap = alpha'*H*alpha + C*sum(max(0,1-H*alpha)) - sum(alpha);
    while times < 10000 && gap>1e-6
        %% 求解子问题
        alphak=alpha;
        for k = 1 : m   
            kk=k;
            Gk = H(kk,:)*alpha-1;
            alpha(kk) = max([min([alpha(kk) - Gk./QQ(kk), upperBound]), lowerBound]);%这里对不对要试试
        end
        times = times + 1;  %外循环次数更新
        Diff = alpha-alphak;
        if mod(times,10 )==0 
            gap = alpha'*H*alpha + C*sum(max(0,1-H*alpha)) - sum(alpha); %更新gap
        end  
%         if times < 10000 %&& norm(Diff) < 1e-6 
%             break;
%         end
    end
    T = toc(tstart);model.x = alpha;model.time=T;model.realnumber=sum(alpha==0)+sum(alpha==C);
    times
end
