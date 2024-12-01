function [alpha, model] = DCDMG1(H, m, C, alpha_int)
    
    alpha = alpha_int;% 产生范围在(lb,ub)的随机数
    tstart=tic;
%     object =  0.5*alpha'*H*alpha -sum(alpha); 
    for k=1:1000 
        order = randperm(m);
        M_max = 0;
        M_min = 0;
        for i = 1:m
            l = order(i);
            G = H(l,:)*alpha-1;
            if alpha(l)==0
                PG = min(G,0);
            elseif alpha(l)==C
                PG = max(G,0);
            else
                PG = G;
            end
            M_max = max(M_max,PG);
            M_min = min(M_min,PG);
            if PG~=0
                alpha(l) = min(max(alpha(l)-G/H(l,l),0),C);
            end    
        end
        k = k+1 ;
        Stop = M_max - M_min;
        object =  0.5*alpha'*H*alpha - sum(alpha);
        Gap =  0.5*alpha'*H*alpha+C*sum(max(0,1-H*alpha)) + object;
        if Gap<= 1e-3  || k==1000 || Stop<1e-6
        break;
        end
    end
    T=toc(tstart);model.time=T;
    model.x = alpha;
    model.object = object;
    k
end

